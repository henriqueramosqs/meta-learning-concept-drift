from frouros.detectors.concept_drift import KSWIN, KSWINConfig
from .mfes_extractor import MfeExtractor
import pandas as pd

class KSWINDetector(MfeExtractor):
    def __init__(self, feature_cols: list =[]):
        """
        Monitors concept drift using KSWIN (Kolmogorov-Smirnov Window) for multiple columns.
        """
        self.feature_cols = feature_cols
        self.detectors = {}
        for col in feature_cols:
            self.detectors[col] = KSWIN()

    def fit(self, data_frame: pd.DataFrame):
        """
        Trains/initializes the detectors with reference data (fills the initial window).
        
        Args:
            data_frame (pd.DataFrame): The reference dataset.
            
        Returns:
            KSWINDetector: The fitted instance (for method chaining).
        """
        for col in self.feature_cols:
            for value in data_frame[col]:
                self.detectors[col].update(value)
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """
        Evaluates new data, updates detectors, and returns metrics + drift flags.
        
        Args:
            data_frame (pd.DataFrame): The new data batch to evaluate.

        Returns:
            dict: Dictionary containing drift flags for each column and a global drift flag.
        """
        results = {}
        drift_detected = False
        
        for col in self.feature_cols:
            col_drift = 0

            for value in data_frame[col]:
                self.detectors[col].update(value)
                if self.detectors[col].drift:
                    drift_detected = True
                    col_drift = 1
            results[f"kswin_{col}_drift"] = col_drift
            
        results["kswin_global_drift_flag"] = drift_detected
        return results
