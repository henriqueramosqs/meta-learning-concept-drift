import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score


METRICS = {"precision","recall","f1-score","kappa"}

class Evaluator():
    def __init__(self):
        pass

    def _get_performance( self, y_true: pd.Series, y_pred: pd.Series, metric_name) -> float:
        metric_dict = {
            "kappa": cohen_kappa_score,
            "r2": r2_score,
            "mse": mean_squared_error,
            "std": lambda y_true, y_pred: np.std(y_true - y_pred),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="micro"),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="micro"),
            "f1-score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
        }
        return metric_dict[metric_name](y_true, y_pred)

    def evaluate(self, metric_name: str, y_true: pd.Series, y_pred: pd.Series,) -> float:
        if metric_name not in METRICS:
            raise ValueError(f"'metric_name' param must be one of {self.metrics}")

        return self._get_performance(y_true, y_pred, metric_name)