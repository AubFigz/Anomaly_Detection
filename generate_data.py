"""
generate_data.py -- synthetic network/security telemetry with REAL injected
anomalies and ground-truth labels, unlike the original generate_test_data.py
(which produced two purely random columns with nothing anomalous to find).
Ground truth is included so detection can be genuinely evaluated (ROC-AUC,
precision/recall), not just run without any way to check correctness.
"""
import numpy as np
import pandas as pd


def generate_security_telemetry(num_records=5000, anomaly_fraction=0.03, seed=42):
    rng = np.random.default_rng(seed)
    n_anomalies = int(num_records * anomaly_fraction)
    n_normal = num_records - n_anomalies

    # Normal traffic: modest packet sizes, steady data rate, few failed logins,
    # a handful of contacted IPs, short-to-moderate connection durations.
    normal = pd.DataFrame({
        "packet_size": rng.normal(500, 120, n_normal).clip(50, 1500),
        "data_rate": rng.normal(20, 6, n_normal).clip(0.5, 100),
        "connection_duration_s": rng.exponential(30, n_normal).clip(1, 300),
        "failed_logins": rng.poisson(0.2, n_normal),
        "unique_ips_contacted": rng.poisson(3, n_normal).clip(1, None),
    })

    # Anomalies: a mix of distinct attack-like patterns -- large data
    # exfiltration bursts, brute-force login attempts, and port-scan-like
    # behavior contacting many IPs briefly. Genuinely different distributions,
    # not just noisier versions of normal.
    kind = rng.choice(["exfiltration", "brute_force", "port_scan"], size=n_anomalies)
    anomalies = pd.DataFrame({
        "packet_size": np.where(kind == "exfiltration", rng.normal(1400, 60, n_anomalies),
                        np.where(kind == "port_scan", rng.normal(80, 20, n_anomalies),
                                 rng.normal(500, 120, n_anomalies))).clip(50, 1500),
        "data_rate": np.where(kind == "exfiltration", rng.normal(85, 8, n_anomalies),
                               rng.normal(15, 8, n_anomalies)).clip(0.5, 100),
        "connection_duration_s": np.where(kind == "exfiltration", rng.normal(250, 30, n_anomalies),
                                  np.where(kind == "port_scan", rng.normal(2, 1, n_anomalies),
                                           rng.normal(15, 8, n_anomalies))).clip(1, 300),
        "failed_logins": np.where(kind == "brute_force", rng.poisson(18, n_anomalies), rng.poisson(0.3, n_anomalies)),
        "unique_ips_contacted": np.where(kind == "port_scan", rng.poisson(40, n_anomalies).clip(15, None),
                                          rng.poisson(3, n_anomalies).clip(1, None)),
    })

    normal["is_anomaly"] = 0
    normal["anomaly_type"] = "normal"
    anomalies["is_anomaly"] = 1
    anomalies["anomaly_type"] = kind

    df = pd.concat([normal, anomalies], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle so anomalies are interspersed
    df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="min")
    return df[["timestamp", "packet_size", "data_rate", "connection_duration_s",
               "failed_logins", "unique_ips_contacted", "is_anomaly", "anomaly_type"]]


if __name__ == "__main__":
    df = generate_security_telemetry()
    df.to_csv("security_telemetry.csv", index=False)
    print(f"Generated {len(df)} records, {df['is_anomaly'].sum()} true anomalies "
          f"({100*df['is_anomaly'].mean():.1f}%)")
    print(df["anomaly_type"].value_counts())
