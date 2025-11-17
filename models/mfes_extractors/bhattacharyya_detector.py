from frouros.detectors.data_drift import BhattacharyyaDistance
from . import MfeExtractor
import pandas as pd
import numpy as np


NUM_BINS = 20
DRIFT_THRESHOLD = 0.15

class BhattacharyyaDetector(MfeExtractor):
    """
    Monitors concept drift using the Bhattacharyya distancE
    for multiple columns.
    
    Args:
        feature_cols (list): List of columns to monitor for drift.
    """
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.first_detectors = {
            col: BhattacharyyaDistance(num_bins=NUM_BINS) for col in feature_cols
        }
        self.last_detectors = {
            col: BhattacharyyaDistance(num_bins=NUM_BINS) for col in feature_cols
        }

    def fit(self, data_frame: pd.DataFrame):
        """
        Initializes the detectors with reference data.
        
        Args:
            data_frame (pd.DataFrame): The reference dataset.
            
        Returns:
            BhattacharyyaDetector: The fitted instance (for method chaining).
        """
        for col in self.feature_cols:
            self.first_detectors[col].fit(data_frame[col].to_numpy())
            self.last_detectors[col].fit(data_frame[col].to_numpy())
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """
        Updates the detectors with a batch of data and returns drift/warning flags and key metrics.

        Args:
            data_frame (pd.DataFrame): The new data batch to evaluate.

        Returns:
            dict: Dictionary containing measured distanes, drift/warning flags for each column and a global drift flag.
        """
        results = {}
        global_drift = False
        
        for col in self.feature_cols:
            distance_result = self.first_detectors[col].compare(data_frame[col].to_numpy())[0][0]
            results[f"bhattacharyya_first_{col}_distance"] = distance_result

            drift_detected = distance_result > DRIFT_THRESHOLD
            results[f"bhattacharyya_first_{col}"] = drift_detected
            if(drift_detected):
                global_drift = True
        
        results["bhattacharyya_global_first_drift_flag"] = global_drift

        global_drift = False

        for col in self.feature_cols:
            distance_result = self.last_detectors[col].compare(data_frame[col].to_numpy())[0][0]
            results[f"bhattacharyya_lst_{col}_distance"] = distance_result

            drift_detected = distance_result > DRIFT_THRESHOLD
            results[f"bhattacharyya_lst_{col}"] = drift_detected
            if(drift_detected):
                global_drift = True
            
            self.last_detectors[col].fit(data_frame[col].to_numpy())

        results["bhattacharyya_global_lst_drift_flag"] = global_drift
        return results
    
if __name__ == "__main__":
    np.random.seed(seed=31)
    
    X = np.random.normal(loc=0, scale=1, size=1000)
    Y = np.random.normal(loc=2, scale=1, size=1000)
    Z = np.random.normal(loc=0.1, scale=1, size=1000)

    df_reference = pd.DataFrame(X, columns=['feature'])
    df_with_drift = pd.DataFrame(Y, columns=['feature'])
    df_without_drift = pd.DataFrame(Z, columns=['feature'])

    detector = BhattacharyyaDetector(feature_cols=['feature'])
    detector.fit(df_reference)

    print("Evaluating test with no drift...")
    results_no_drift = detector.evaluate(df_without_drift)
    print(results_no_drift)
    print("-" * 50)


    print("Evaluating test with with drift...")
    results_drift = detector.evaluate(df_with_drift)
    print(results_drift)