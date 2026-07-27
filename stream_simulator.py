"""
stream_simulator.py -- simulates real-time batch anomaly detection over the
generated telemetry, updating real Prometheus metrics via prometheus_client
(genuinely queryable at /metrics, not mocked) and logging alerts.

This replaces the original's Kafka + AWS SNS pipeline, which needs real
external infrastructure this environment doesn't have. It's an honest
stand-in for demonstrating the same detection logic end-to-end locally --
see NOTES.md for exactly what's simulated here vs. what the original
architecture (Kafka streaming, AWS Lambda/GCP deployment, YOLO CCTV
integration, Rasa chatbot) would need for a full production version.
"""
import logging
import time

from prometheus_client import Gauge, Counter, start_http_server

from detector import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

anomaly_gauge = Gauge("anomaly_count_current_batch", "Anomalies detected in the most recent batch")
total_anomalies_counter = Counter("anomalies_detected_total", "Total anomalies detected since start")
latency_gauge = Gauge("model_inference_latency_seconds", "Latency of the most recent batch's model inference")
batches_processed_counter = Counter("batches_processed_total", "Total batches processed")

_alerts_log = []  # in-memory recent-alerts list, read by the Flask dashboard


def send_alert(message):
    """Stands in for the original's AWS SNS call -- logs locally instead of
    publishing to a real topic, since no AWS account is configured here."""
    logging.info(f"ALERT: {message}")
    _alerts_log.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message})
    if len(_alerts_log) > 50:
        _alerts_log.pop(0)


def get_recent_alerts():
    return list(reversed(_alerts_log))


def run_batch_detection(df, model, scaler, batch_size=25, delay=0.05):
    """Processes the dataframe in batches, as if it were arriving from a
    live stream, updating real Prometheus metrics and logging alerts for
    any batch containing a detected anomaly."""
    total_detected = 0
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        t0 = time.time()
        X_batch, _ = preprocess(batch, scaler=scaler, fit=False)
        preds = model.predict(X_batch)
        n_anomalies = int((preds == -1).sum())
        latency = time.time() - t0

        anomaly_gauge.set(n_anomalies)
        latency_gauge.set(latency)
        batches_processed_counter.inc()
        if n_anomalies:
            total_anomalies_counter.inc(n_anomalies)
            total_detected += n_anomalies
            anomalous_rows = batch[preds == -1]
            send_alert(f"{n_anomalies} anomaly(ies) detected in batch starting row {start} "
                       f"(sample packet_size={anomalous_rows['packet_size'].iloc[0]:.0f}, "
                       f"data_rate={anomalous_rows['data_rate'].iloc[0]:.1f})")
        time.sleep(delay)
    return total_detected


if __name__ == "__main__":
    import pickle
    from generate_data import generate_security_telemetry

    start_http_server(8000)
    logging.info("Prometheus metrics available at http://localhost:8000/metrics")

    df = generate_security_telemetry()
    with open("saved_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("saved_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    logging.info(f"Streaming {len(df)} records in batches of 25...")
    total = run_batch_detection(df, model, scaler)
    logging.info(f"Done. {total} anomalies detected across the simulated stream.")
