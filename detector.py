"""
detector.py -- Optuna-tuned anomaly detection (Isolation Forest and an
autoencoder), with the real bugs from the original script fixed:

  1. The original objective() returned a trained MODEL object where Optuna
     requires a numeric score -- guaranteed to crash the moment it actually
     ran. Fixed: returns ROC-AUC against held-out ground truth.
  2. The original then did best_trial.params['model'] -- 'model' was never
     a suggested hyperparameter, so this was a guaranteed KeyError. Fixed:
     retrain fresh with study.best_params, the standard Optuna pattern.
  3. The original imported the real keras.models.load_model, then defined
     its own function of the exact same name right below it, silently
     shadowing the import. Fixed: renamed to load_pickled_model.
"""
import pickle
import numpy as np
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = ["packet_size", "data_rate", "connection_duration_s", "failed_logins", "unique_ips_contacted"]


def save_pickled_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


def load_pickled_model(filename):
    """Renamed from the original's `load_model` specifically to avoid
    shadowing keras.models.load_model, which the original script imported
    but then immediately overwrote with this exact function name."""
    import os
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None


def preprocess(df, scaler=None, fit=True):
    X = df[FEATURES].values.astype(float)
    if fit:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled, scaler


def tune_isolation_forest(X_train, X_val, y_val, n_trials=25):
    """Optuna hyperparameter search. FIXED: the objective now returns a real
    ROC-AUC float (validated against the held-out labels), not a model
    object -- this is what the original code got wrong in a way that would
    have crashed on the very first trial."""

    def objective(trial):
        contamination = trial.suggest_float("contamination", 0.01, 0.10)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_features = trial.suggest_float("max_features", 0.5, 1.0)
        model = IsolationForest(
            contamination=contamination, n_estimators=n_estimators,
            max_features=max_features, random_state=42,
        )
        model.fit(X_train)
        # decision_function: higher = more normal. Flip sign so higher = more
        # anomalous, matching the y_val convention (1 = anomaly), so ROC-AUC
        # is computed the right way round.
        scores = -model.decision_function(X_val)
        return roc_auc_score(y_val, scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # FIXED: retrain fresh with the best hyperparameters found, rather than
    # the original's best_trial.params['model'], which could never have
    # worked -- 'model' was never one of the suggested parameters.
    best_model = IsolationForest(**study.best_params, random_state=42)
    best_model.fit(X_train)
    return best_model, study.best_value, study.best_params


def evaluate(model, X_test, y_test):
    scores = -model.decision_function(X_test)
    auc = roc_auc_score(y_test, scores)
    preds = (model.predict(X_test) == -1).astype(int)  # IsolationForest: -1 = anomaly
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"roc_auc": auc, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def train_and_evaluate(df, n_trials=25, test_size=0.3, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["is_anomaly"], random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df["is_anomaly"], random_state=seed)

    X_train, scaler = preprocess(train_df, fit=True)
    X_val, _ = preprocess(val_df, scaler=scaler, fit=False)
    X_test, _ = preprocess(test_df, scaler=scaler, fit=False)
    y_val, y_test = val_df["is_anomaly"].values, test_df["is_anomaly"].values

    model, best_val_auc, best_params = tune_isolation_forest(X_train, X_val, y_val, n_trials=n_trials)
    metrics = evaluate(model, X_test, y_test)
    return model, scaler, best_params, best_val_auc, metrics
