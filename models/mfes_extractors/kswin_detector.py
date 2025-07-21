from frouros.detectors.concept_drift import KSWIN, KSWINConfig
import pandas as pd

class KSWINDetector:
    def __init__(self, feature_cols: list, window_size: int = 100, alpha: float = 0.05):
        """Inicializa detectores KSWIN para múltiplas colunas.
        
A
            feature_cols (list): Lista de nomes de colunas para monitorar
            window_size (int): Tamanho da janela para o teste KSWIN
            alpha (float): Nível de significância para detecção de drift
        """
        self.feature_cols = feature_cols
        self.detectors = {}
        
        for col in feature_cols:
            self.detectors[col] = KSWIN()

    def fit(self, data_frame: pd.DataFrame):
        """Treina os detectores com dados de referência."""
        for col in self.feature_cols:
            for value in data_frame[col]:
                self._update_detector(col, value)
        return self

    def evaluate(self, data_frame: pd.DataFrame) -> dict:
        """Avalia novos dados e retorna métricas + flag de drift."""
        results = {}
        global_drift = 0
        drift_detected = False
        
        for col in self.feature_cols:
            col_drift = 0

            for value in data_frame[col]:
                drift_status = self._update_detector(col, value)
                self.detectors[col].update(value)
                if self.detectors[col].drift:
                    drift_detected = True
                    col_drift = 1
            results[f"kswin_{col}_drift"] = col_drift
            
        if(drift_detected):
                global_drift=True
            
        results["kswin_global_drift_flag"] = global_drift
        return results

    def _update_detector(self, col: str, value: float):
        """Método interno para atualizar o detector de forma compatível."""
        try:
            return self.detectors[col].update(value)
        except Exception as e:
            print(f"Warning: Error updating detector for column {col}: {str(e)}")
            return {}