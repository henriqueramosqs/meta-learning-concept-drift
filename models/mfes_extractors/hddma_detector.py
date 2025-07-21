from capymoa.drift.detectors import HDDMAverage
import pandas as pd

class HDDMADetector:
    """Monitora drift usando HDDM-A para múltiplas colunas.
    
    Args:
        feature_cols (list): Lista de colunas a serem monitoradas.
        hddma_params (dict): Parâmetros do HDDM-A (ex: drift_confidence=0.001).
    """
    def __init__(self, feature_cols: list, hddma_params: dict = {}):
        self.feature_cols = feature_cols
        self.detectors = {
            col: HDDMAverage(**hddma_params) for col in feature_cols
        }

    def fit(self, data_frame: pd.DataFrame):
        """Inicializa os detectores com dados de referência."""
        for col in self.feature_cols:
            for value in data_frame[col]:
                self.detectors[col].add_element(value)
        return self

    def _check_drift(self, data_frame: pd.DataFrame) -> int:
        """Verifica se qualquer coluna detectou drift."""
        for col in self.feature_cols:
            for value in data_frame[col]:
                self.detectors[col].add_element(value)
                if self.detectors[col].detected_change():
                    return 1
        return 0

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """Retorna métricas + flag de drift."""
        results = {}
        drift_detected = False
        
        for col in self.feature_cols:
            col_drift = False
            for value in data_frame[col]:
                self.detectors[col].add_element(value)
                if self.detectors[col].detected_change():
                    col_drift = True
            results[f"hddma_{col}_drift"] = int(col_drift)
            results[f"hddma_{col}_warning"] = int(self.detectors[col].detected_warning())
            
            if col_drift:
                drift_detected = True
        
        results["hddma_global_drift_flag"] = int(drift_detected)
        return results