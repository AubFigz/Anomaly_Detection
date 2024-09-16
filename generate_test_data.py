import pandas as pd
import numpy as np

# Function to generate synthetic network traffic data
def generate_network_traffic_data(num_records=1000):
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=num_records, freq="T"),
        "packet_size": np.random.randint(50, 1500, size=num_records),
        "data_rate": np.random.random(size=num_records) * 100
    }
    df = pd.DataFrame(data)
    df.to_csv("test_network_traffic_data.csv", index=False)
    print(f"Generated {num_records} records of network traffic data.")

if __name__ == "__main__":
    generate_network_traffic_data()
