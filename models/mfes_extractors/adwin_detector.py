from frouros.detectors.concept_drift import ADWIN
import pandas as pd

class ADWINDetector():
    """Monitors concept drift using ADWIN for multiple columns.

        Args:
            feature_cols (list): List of columns to monitor for drift.
            adwin_params (dict): ADWIN parameters (e.g., delta=0.002).
    """

    
    def __init__(self, feature_cols: list=[], adwin_params: dict = {}):
        """ Creates an ADWIN detector instance for each feature column"""
        self.feature_cols = feature_cols
        self.detectors = {
            col: ADWIN(**adwin_params) for col in feature_cols
        }

    def fit(self, data_frame: pd.DataFrame):
        """Initializes the ADWIN detectors with reference data.

        Args:
            data_frame (pd.DataFrame): The reference dataset.
        
        Returns:
            ADWINDetector: The fitted instance (for method chaining).
        """
        for col in self.feature_cols:
            for value in data_frame[col]:
                self.detectors[col].update(value)
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """Updates the detectors and returns metrics plus a drift flag.

        Args:
            data_frame (pd.DataFrame): The new data batch to evaluate.

        Returns:
            dict: Dictionary containing ADWIN-specific metrics (like window width)
                  and the overall drift flag.
        """
          
        results = {}
        drift_detected = False
        print("FEATURE_COLS",self.feature_cols)
        for col in self.feature_cols:
            for value in data_frame[col]:
                self.detectors[col].update(value)
                if self.detectors[col].drift:
                    drift_detected = True
            results[f"adwin_{col}"] = self.detectors[col].width  

        results["adwin_drift_flag"] = int(drift_detected)
        return results

