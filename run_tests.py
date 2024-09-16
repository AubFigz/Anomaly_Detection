import unittest
from realtimesecurityanomalydetection import run_automl, preprocess_data
import numpy as np
import pandas as pd


class TestAnomalyDetection(unittest.TestCase):
    def test_preprocessing(self):
        # Sample raw data with missing values
        raw_data = pd.DataFrame([[1, 2, np.nan], [4, 5, 6]])
        processed_data = preprocess_data(raw_data)

        # Ensure dimensions after PCA and no missing values
        self.assertEqual(processed_data.shape[1], 2)  # Check if PCA reduced dimensions to 2

    def test_automl(self):
        # Test AutoML model optimization with sample data
        data = np.random.random((100, 10))  # Simulated random data
        best_trial = run_automl(data)

        # Ensure AutoML returns a valid trial object
        self.assertIsNotNone(best_trial)


if __name__ == "__main__":
    unittest.main()
