"""
run_demo.py -- trains the model if needed, starts the Prometheus metrics
server and the Flask dashboard, and continuously loops the batch-detection
simulation in a background thread so the dashboard has live, changing data
to show.
"""
import logging
import pickle
import threading
import time

from prometheus_client import start_http_server

from generate_data import generate_security_telemetry
from detector import train_and_evaluate, save_pickled_model, load_pickled_model
from stream_simulator import run_batch_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_or_train_model():
    model = load_pickled_model("saved_model.pkl")
    scaler = load_pickled_model("saved_scaler.pkl")
    if model is not None and scaler is not None:
        logging.info("Loaded existing trained model.")
        return model, scaler
    logging.info("No saved model found -- training now (Optuna, 25 trials)...")
    df = generate_security_telemetry()
    model, scaler, best_params, best_val_auc, metrics = train_and_evaluate(df, n_trials=25)
    logging.info(f"Trained. Best params: {best_params}")
    logging.info(f"Validation ROC-AUC: {best_val_auc:.4f} | Test metrics: {metrics}")
    save_pickled_model(model, "saved_model.pkl")
    save_pickled_model(scaler, "saved_scaler.pkl")
    return model, scaler


def continuous_stream_loop(model, scaler):
    """Keeps generating fresh batches of traffic and running detection on
    them forever, so the dashboard always has live-looking data."""
    while True:
        df = generate_security_telemetry(num_records=1000, seed=int(time.time()) % 10000)
        run_batch_detection(df, model, scaler, batch_size=20, delay=0.3)


if __name__ == "__main__":
    model, scaler = get_or_train_model()

    start_http_server(8000)
    logging.info("Prometheus metrics: http://localhost:8000/metrics")

    threading.Thread(target=continuous_stream_loop, args=(model, scaler), daemon=True).start()
    logging.info("Background detection loop started.")

    from dashboard import app
    logging.info("Dashboard: http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
