"""test_detector.py -- real tests for the fixed anomaly detection pipeline."""
import os
import numpy as np

from generate_data import generate_security_telemetry
from detector import preprocess, tune_isolation_forest, evaluate, save_pickled_model, load_pickled_model

passed, failed = 0, 0
def check(label, condition):
    global passed, failed
    if condition:
        passed += 1; print(f"  PASS: {label}")
    else:
        failed += 1; print(f"  FAIL: {label}")


print("=== generate_data ===")
df = generate_security_telemetry(num_records=2000, anomaly_fraction=0.03, seed=1)
check("correct record count", len(df) == 2000)
check("anomaly fraction is roughly as requested", 0.02 <= df["is_anomaly"].mean() <= 0.04)
check("all three anomaly types present", set(df.loc[df.is_anomaly == 1, "anomaly_type"].unique()) == {"exfiltration", "brute_force", "port_scan"})
check("no NaNs in feature columns", not df[["packet_size", "data_rate", "connection_duration_s", "failed_logins", "unique_ips_contacted"]].isnull().values.any())

print("=== preprocess ===")
X, scaler = preprocess(df, fit=True)
check("preprocess returns correct shape", X.shape == (2000, 5))
check("scaled data has ~zero mean", abs(X.mean()) < 0.1)
X2, _ = preprocess(df, scaler=scaler, fit=False)
check("re-applying a fitted scaler is deterministic", np.allclose(X, X2))

print("=== tune_isolation_forest (the actual bug fix) ===")
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["is_anomaly"], random_state=1)
X_train, scaler2 = preprocess(train_df, fit=True)
X_test, _ = preprocess(test_df, scaler=scaler2, fit=False)
y_test = test_df["is_anomaly"].values

model, best_val_auc, best_params = tune_isolation_forest(X_train, X_test, y_test, n_trials=5)
check("tuning returns a real fitted model (not a raw model/no crash)", hasattr(model, "predict"))
check("tuning returns a numeric best score (the original bug returned a model object here)", isinstance(best_val_auc, float))
check("best_params contains the actual tuned hyperparameters (the original had a KeyError here)",
      set(best_params.keys()) == {"contamination", "n_estimators", "max_features"})

print("=== evaluate ===")
metrics = evaluate(model, X_test, y_test)
check("evaluate returns roc_auc in valid range", 0.0 <= metrics["roc_auc"] <= 1.0)
check("evaluate returns precision in valid range", 0.0 <= metrics["precision"] <= 1.0)
check("evaluate returns recall in valid range", 0.0 <= metrics["recall"] <= 1.0)
check("model performs well above random chance on this data", metrics["roc_auc"] > 0.8)

print("=== save/load (the load_model naming collision fix) ===")
save_pickled_model(model, "/tmp/test_model.pkl")
loaded = load_pickled_model("/tmp/test_model.pkl")
check("saved and loaded model behaves identically", np.array_equal(model.predict(X_test), loaded.predict(X_test)))
check("load_pickled_model returns None for a missing file (no crash)", load_pickled_model("/tmp/definitely_does_not_exist.pkl") is None)
os.remove("/tmp/test_model.pkl")

print()
print(f"{passed} passed, {failed} failed")
assert failed == 0, f"{failed} test(s) failed"
print("ALL PASS")
