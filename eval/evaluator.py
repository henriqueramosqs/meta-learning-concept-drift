import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score


METRICS = {"precision","recall","f1-score","kappa"}

class Evaluator():
    """
    A utility class designed to centralize the calculation of various 
    performance metrics
    """

    def __init__(self,evaluator_avg="micro"):

        """
        Initializes the Evaluator with a specified averaging method for 
        multi-class classification metrics.

        Args:
            evaluator_avg (str): Averaging method for classification metrics 
                                 like precision/recall/f1 (e.g., 'micro', 'macro', 'weighted'). 
                                 Defaults to "micro".
        """

        self.evaluator_avg = evaluator_avg
        if(self.evaluator_avg==None):
            self.evaluator_avg = "micro"
        print(f"evaluator: {self.evaluator_avg}")
        
    def _get_performance( self, y_true: pd.Series, y_pred: pd.Series, metric_name) -> float:

        """
        Defines the dictionary of available performance metrics and their 
        corresponding functions callable functinos
        
        Returns:
            Dict: Mapping of metric name (str) to its calculation function.
        """
        metric_dict = {
            "kappa": cohen_kappa_score,
            "r2": r2_score,
            "mse": mean_squared_error,
            "std": lambda y_true, y_pred: np.std(y_true - y_pred),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred
                                                                ,average=self.evaluator_avg
                                                                ),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred
                                                          , average=self.evaluator_avg
                                                          ),
            "f1-score": lambda y_true, y_pred: f1_score(y_true, y_pred
                                                        , average=self.evaluator_avg
                                                        ),
        }
        return metric_dict[metric_name](y_true, y_pred)

    def evaluate(self, metric_name: str, y_true: pd.Series, y_pred: pd.Series,) -> float:
        """
        Calculates a specified performance metric between true and predicted values.
        
        Args:
            metric_name (str): The name of the metric to calculate (e.g., 'f1-score', 'r2').
            y_true (pd.Series): The ground truth target values.
            y_pred (pd.Series): The model's predicted target values.

        Returns:
            float: The calculated metric score.
        """

        if metric_name not in METRICS:
            raise ValueError(f"'metric_name' param must be one of {self.metrics}")
        return self._get_performance(y_true, y_pred, metric_name)