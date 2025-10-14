from capymoa.drift.detectors import HDDMWeighted
from .mfes_extractor import MfeExtractor
import pandas as pd

class HDDMWDetector(MfeExtractor):
    """
    Monitors concept drift using the HDDM-W (Hellinger Distance Drift Detection Method - Weighted) 
    for multiple columns. HDDM-W uses a weighted moving average.
    
    Args:
        feature_cols (list): List of columns to monitor for drift.
        hddmw_params (dict): Parameters for HDDM-W (e.g., drift_confidence=0.001).
    """
    def __init__(self, feature_cols: list, hddmw_params: dict = {}):
        self.feature_cols = feature_cols
        self.detectors = {
            col: HDDMWeighted(**hddmw_params) for col in feature_cols
        }

    def fit(self, data_frame: pd.DataFrame):
        """
        Initializes the detectors with reference data.
        
        Args:
            data_frame (pd.DataFrame): The reference dataset.
            
        Returns:
            HDDMWDetector: The fitted instance (for method chaining).
        """

        for col in self.feature_cols:
            for value in data_frame[col]:
                self.detectors[col].add_element(value)
        return self
    

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """
        Updates the detectors with a batch of data and returns drift/warning flags and key metrics.

        Args:
            data_frame (pd.DataFrame): The new data batch to evaluate.

        Returns:
            dict: Dictionary containing drift/warning flags for each column and a global drift flag.
        """
        results = {}
        drift_detected = False
        
        for col in self.feature_cols:
            col_drift = False
            for value in data_frame[col]:
                self.detectors[col].add_element(value)
                if self.detectors[col].detected_change():
                    col_drift = True
            results[f"hddmw_{col}_drift"] = int(col_drift)
            results[f"hddmw_{col}_warning"] = int(self.detectors[col].detected_warning())
            
            if col_drift:
                drift_detected = True
        
        results["hddmw_global_drift_flag"] = int(drift_detected)
        return results