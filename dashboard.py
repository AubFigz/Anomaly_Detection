"""
dashboard.py -- a working Flask dashboard for the anomaly detection system,
replacing the original create_flask_dashboard() (which only had bare JSON
stub endpoints with no real data wired in, not something to screenshot).
"""
from flask import Flask, jsonify, render_template_string

from stream_simulator import anomaly_gauge, latency_gauge, batches_processed_counter, get_recent_alerts

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Anomaly Detection Dashboard</title>
<meta http-equiv="refresh" content="3">
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; background: #0f1720; color: #e5e9ec; margin: 0; padding: 32px; }
  h1 { color: #fff; font-size: 22px; }
  .cards { display: flex; gap: 16px; margin: 24px 0; }
  .card { background: #1a2634; border-radius: 10px; padding: 20px 28px; flex: 1; border: 1px solid #2a3a4a; }
  .card .label { color: #8ea0ad; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 32px; font-weight: 700; margin-top: 6px; }
  .value.alert { color: #ff5d6c; }
  .value.ok { color: #4ade80; }
  table { width: 100%; border-collapse: collapse; margin-top: 12px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a3a4a; font-size: 13px; }
  th { color: #8ea0ad; text-transform: uppercase; font-size: 11px; }
  .badge { background: #ff5d6c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; }
</style>
</head>
<body>
  <h1>&#128737; Real-Time Security Anomaly Detection</h1>
  <div class="cards">
    <div class="card">
      <div class="label">Current batch anomalies</div>
      <div class="value {{ 'alert' if current_anomalies > 0 else 'ok' }}">{{ current_anomalies }}</div>
    </div>
    <div class="card">
      <div class="label">Total anomalies detected</div>
      <div class="value alert">{{ total_anomalies }}</div>
    </div>
    <div class="card">
      <div class="label">Batches processed</div>
      <div class="value ok">{{ batches_processed }}</div>
    </div>
    <div class="card">
      <div class="label">Last batch latency</div>
      <div class="value ok">{{ latency_ms }} ms</div>
    </div>
  </div>
  <h2>Recent Alerts <span class="badge">{{ alerts|length }}</span></h2>
  <table>
    <tr><th>Time</th><th>Message</th></tr>
    {% for a in alerts %}
    <tr><td>{{ a.time }}</td><td>{{ a.message }}</td></tr>
    {% endfor %}
  </table>
  <p style="color:#5c6b78;font-size:12px;margin-top:24px;">
    Metrics also available in Prometheus format at <a style="color:#8ea0ad" href="/metrics-json">/metrics-json</a>
    (this local demo exposes plain JSON; the full prometheus_client /metrics endpoint runs on port 8000).
  </p>
</body>
</html>
"""


@app.route("/")
def index():
    alerts = get_recent_alerts()[:15]
    return render_template_string(
        PAGE,
        current_anomalies=int(anomaly_gauge._value.get()),
        total_anomalies=_total_from_alerts(),
        batches_processed=int(batches_processed_counter._value.get()),
        latency_ms=round(latency_gauge._value.get() * 1000, 1),
        alerts=alerts,
    )


def _total_from_alerts():
    import re
    total = 0
    for a in get_recent_alerts():
        m = re.match(r"(\d+) anomaly", a["message"])
        if m:
            total += int(m.group(1))
    return total


@app.route("/metrics-json")
def metrics_json():
    return jsonify({
        "current_batch_anomalies": int(anomaly_gauge._value.get()),
        "batches_processed": int(batches_processed_counter._value.get()),
        "last_latency_seconds": latency_gauge._value.get(),
        "recent_alerts": get_recent_alerts()[:10],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
