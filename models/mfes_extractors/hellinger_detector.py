from frouros.detectors.data_drift import HellingerDistance 
                                            
from . import MfeExtractor  
import pandas as pd                        
import numpy as np                        
DRIFT_THRESHOLD = 0.15

class HellingerDistanceDetector(MfeExtractor):
    
# class HellingerDistanceDetector():
    """
    Monitors concept drift using the Earth Mover's Distance
    for multiple columns.
    
    Args:
        feature_cols (list): List of columns to monitor for drift.
    """
    def __init__(self, feature_cols: list):
        """
        Initializes the detector.
        
        Args:
            feature_cols (list): A list of column names (features) to monitor for distribution drift.
        """
        self.feature_cols = feature_cols
        self.first_detectors = {
            col: HellingerDistance() for col in feature_cols
        }

        self.last_detectors = {
            col: HellingerDistance() for col in feature_cols
        }

    def fit(self, data_frame: pd.DataFrame):
        """
        Initializes and fits the detectors with a reference dataset.
        This dataset represents the "normal" or expected distribution for each feature.
        
        Args:
            data_frame (pd.DataFrame): The reference dataset (e.g., training data).
            
        Returns:
            HellingerDistanceDetector: The fitted instance itself, to allow for method chaining
        """
        for col in self.feature_cols:
            self.first_detectors[col].fit(data_frame[col].to_numpy())
            self.last_detectors[col].fit(data_frame[col].to_numpy())
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """
        Compares a new batch of data to the reference data to detect drift.
        It updates the detectors and returns drift flags and key metrics.

        Args:
            data_frame (pd.DataFrame): The new data batch to evaluate for drift.

        Returns:
            dict: A dictionary containing the measured distance, a boolean drift flag for each column,
                  and a single global drift flag that is true if any column has drifted.
        """
        results = {}
        global_drift = False
        
        for col in self.feature_cols:
            distance_result = self.first_detectors[col].compare(data_frame[col].to_numpy())[0][0]
            results[f"hellinger_distance_first_{col}_distance"] = distance_result
            drift_detected = distance_result > DRIFT_THRESHOLD
            results[f"hellinger_distance_first_{col}"] = drift_detected
            if(drift_detected):
                global_drift = True
        
        results["hellinger_global_first_drift_flag"] = global_drift

        global_drift = False
        
        for col in self.feature_cols:
            distance_result = self.last_detectors[col].compare(data_frame[col].to_numpy())[0][0]
            results[f"hellinger_distance_lst_{col}_distance"] = distance_result

            drift_detected = distance_result > DRIFT_THRESHOLD
            results[f"hellinger_distance_lst_{col}"] = drift_detected
            if(drift_detected):
                global_drift = True
            self.last_detectors[col].fit(data_frame[col].to_numpy())
        
        results["hellinger_global_lst_drift_flag"] = global_drift
        return results
    
if __name__ == "__main__":

    np.random.seed(seed=31)
    
    X = np.random.normal(loc=0, scale=1, size=1000)
    Y = np.random.normal(loc=2, scale=1, size=1000)
    Z = np.random.normal(loc=0.1, scale=1, size=1000)

    df_reference = pd.DataFrame(X, columns=['feature'])
    df_with_drift = pd.DataFrame(Y, columns=['feature'])
    df_without_drift = pd.DataFrame(Z, columns=['feature'])

    detector = HellingerDistanceDetector(feature_cols=['feature'])
    detector.fit(df_reference)

    print("Evaluating test with no drift...")
    results_no_drift = detector.evaluate(df_without_drift)
    print(results_no_drift)
    print("-" * 50)


    print("Evaluating test with with drift...")
    results_drift = detector.evaluate(df_with_drift)
    print(results_drift)