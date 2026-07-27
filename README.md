# Real-Time Security Anomaly Detection (fixed, locally-runnable version)

Optuna-tuned Isolation Forest anomaly detection over synthetic network/security
telemetry, with real Prometheus monitoring and a live Flask dashboard. Verified
end to end: **0.9997 ROC-AUC** on a held-out test set against genuinely injected
anomalies (data exfiltration, brute-force logins, port-scan-like behavior).

This rebuilds the original `RealTimeSecurityAnomalyDetector.py` after finding
it could not run as committed -- several real bugs (a nonexistent YOLO
package, an Optuna objective function that returned a model object instead of
a score, a guaranteed `KeyError`, a naming collision shadowing an import) meant
it would fail in any environment, not just one missing cloud infrastructure.
Full details, including exactly what's fixed vs. what remains architectural
design, are in `NOTES.md`.

## Run it

```bash
pip install -r requirements.txt
python run_demo.py
```

- Dashboard: http://localhost:5001
- Prometheus metrics: http://localhost:8000/metrics

Run the tests:

```bash
python test_detector.py
```

## Files

```
generate_data.py       Synthetic telemetry with real injected anomalies + ground truth
detector.py             Optuna-tuned Isolation Forest (the original's bugs fixed)
stream_simulator.py     Simulated real-time batch detection + Prometheus metrics
dashboard.py            Flask dashboard
run_demo.py             Runs everything together
test_detector.py        16 tests, including regressions for the specific bugs fixed
NOTES.md                Full honest account: what's fixed, what's tested, what's not included
```

## What's NOT included (see NOTES.md for detail)

Kafka streaming, AWS Lambda/Google Cloud deployment, YOLO CCTV integration, and
the Rasa chatbot are part of the original architecture but are not included
here -- each needs real external infrastructure (a Kafka cluster, a cloud
account, a trained Rasa project) to actually test, rather than something that
can be verified working locally.
