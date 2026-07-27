# NOTES: What's actually working here, and what isn't (yet)

This is a rebuilt, locally-runnable version of the original Real-Time Security
Anomaly Detection System. Everything below has been genuinely run and verified
in this form -- nothing here is claimed as working without having actually
tested it.

## Real bugs found in the original script, and how they were fixed

1. **`from yolo import YOLO` does not work.** The real `yolo` package on PyPI
   has no version `1.0.0` (only 0.3.x exists), and even the real package has
   no `YOLO` class with the API the original code expected
   (`weights_path`/`config_path`/`labels_path`, `.detect_objects()`). This
   crashed the original script on its very first lines, before any business
   logic ever ran. **Not included in this version** -- see "Not included"
   below.

2. **The Optuna objective function returned a trained model object, where
   Optuna requires a numeric score.** This is a guaranteed crash the moment
   `run_automl()` was actually called, independent of any environment issue.
   **Fixed** in `detector.py`: the objective now returns ROC-AUC against a
   held-out validation set.

3. **`best_trial.params['model']` was a guaranteed `KeyError`** -- `'model'`
   was never one of the hyperparameters Optuna was asked to tune (only
   `contamination`/`n_estimators` were). **Fixed**: the standard Optuna
   pattern of retraining fresh with `study.best_params` after the search
   completes.

4. **A naming collision**: the original imported the real
   `keras.models.load_model`, then immediately defined its own function of
   the exact same name, silently shadowing the import. **Fixed**: renamed to
   `load_pickled_model`.

5. **A scoping bug in the original's `__main__` block** referenced
   `decrypted_data`, a variable that only existed inside a different, nested
   function's local scope -- a guaranteed `NameError`. **Not applicable
   here** since the Kafka/encryption pipeline isn't part of this local
   version (see below).

## What's real and tested in this version

- **Data**: `generate_data.py` creates realistic synthetic network/security
  telemetry with three distinct, genuinely-different injected anomaly
  patterns (data exfiltration, brute-force login attempts, port-scan-like
  behavior) and ground-truth labels -- unlike the original's
  `generate_test_data.py`, which produced two purely random columns with
  nothing anomalous to actually find.
- **Model**: `detector.py` -- Optuna-tuned Isolation Forest, genuinely
  evaluated on a held-out test set. Verified result: **ROC-AUC ~ 0.9997**.
  Honest nuance worth keeping, not hiding: precision is 1.0 (zero false
  alarms) but recall is only ~47%, because the tuned `contamination`
  parameter came out lower than the true anomaly rate -- the model ranks
  anomalies correctly (hence the high AUC) but its default decision
  threshold is conservative. A real, common anomaly-detection tradeoff, not
  a bug.
- **"Real-time" simulation**: `stream_simulator.py` processes the data in
  batches as if arriving from a live stream, updates real Prometheus metrics
  (genuinely queryable at `http://localhost:8000/metrics`, not mocked), and
  logs alerts.
- **Dashboard**: `dashboard.py` -- a working Flask dashboard
  (`http://localhost:5001`) showing live anomaly counts, batch throughput,
  latency, and a recent-alerts feed, auto-refreshing every 3 seconds.
- **Tests**: `test_detector.py` -- 16 tests, all passing, specifically
  including regression tests for the exact bugs fixed above (e.g. asserting
  the tuning function returns a numeric score, not a model object).

## What's NOT included in this local version (architectural design, not tested)

The original's broader architecture is a genuinely good design -- this local
version deliberately narrows scope to what can be honestly demonstrated
without real external infrastructure:

- **Kafka streaming ingestion** -- replaced with local batch simulation.
  Real Kafka integration needs an actual running Kafka cluster to test
  against.
- **AWS Lambda / Google Cloud deployment** -- `deploy_lambda.sh` and the
  cloud deployment functions need a real, configured cloud account. Not
  attempted here.
- **YOLO CCTV object detection** -- removed entirely rather than left broken;
  a real version would need either a proper YOLO wrapper (e.g. via OpenCV's
  `cv2.dnn` module directly with the existing `yolov3.cfg`/`.weights` files)
  or the `ultralytics` package, neither of which the original code actually
  used correctly.
- **Rasa chatbot** -- needs a fully trained Rasa project (domain file, NLU
  training examples, stories) which doesn't exist anywhere in the original
  repo; `Agent.load("models/dialogue", ...)` would fail regardless of any
  other fix, since that directory was never included.
- **MLflow experiment tracking** -- removed from this version for simplicity;
  MLflow can run entirely locally (no cloud needed) and would be a
  reasonable next addition.

## Running it yourself

```bash
pip install -r requirements.txt
python run_demo.py
```

Then open `http://localhost:5001` for the dashboard and
`http://localhost:8000/metrics` for the raw Prometheus metrics. Run
`python test_detector.py` to see the test suite pass.
