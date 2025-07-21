from frouros.detectors.concept_drift import ADWIN
import pandas as pd

class ADWINDetector():
    """Monitora drift usando ADWIN para múltiplas colunas.
    
    Args:
        feature_cols (list): Lista de colunas a serem monitoradas.
        adwin_params (dict): Parâmetros do ADWIN (ex: delta=0.002).
    """
    def __init__(self, feature_cols: list, adwin_params: dict = {}):
        self.feature_cols = feature_cols
        self.detectors = {
            col: ADWIN(**adwin_params) for col in feature_cols
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
                self.detectors[col].update(value)
                if self.detectors[col].drift:
                    return 1
        return 0

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """Retorna métricas + flag de drift."""
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

