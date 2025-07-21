from capymoa.drift.detectors import HDDMWeighted
import pandas as pd

class HDDMWDetector:
    """Monitora drift usando HDDM-W (Wilcoxon) para múltiplas colunas.
    
    Args:
        feature_cols (list): Lista de colunas a serem monitoradas.
        hddmw_params (dict): Parâmetros do HDDM-W (ex: drift_confidence=0.001).
    """
    def __init__(self, feature_cols: list, hddmw_params: dict = {}):
        self.feature_cols = feature_cols
        self.detectors = {
            col: HDDM_W(**hddmw_params) for col in feature_cols
        }

    def fit(self, data_frame: pd.DataFrame):
        """Inicializa os detectores com dados de referência."""
        for col in self.feature_cols:
            for value in data_frame[col]:
                self.detectors[col].update(value)
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
                self.detectors[col].update(value)
                if self.detectors[col].detected_change():
                    col_drift = True
            results[f"hddmw_{col}_drift"] = int(col_drift)
            results[f"hddmw_{col}_median"] = self.detectors[col].get_median_estimation()
            
            if col_drift:
                drift_detected = True
        
        results["hddmw_global_drift_flag"] = int(drift_detected)
        return results