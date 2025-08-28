import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from models import Evaluator
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data.utils.eda import EDA

# Macros
COLORS = [
    "#eb5600ff", # orange
    "#1a9988ff", # green
    "#595959ff", # grey
    "#6aa4c8ff", # blue
    "#f1c232ff", # yellow
    ]
BASE_MODELS = [
    "RandomForestClassifier",
    "SVC",
    "LogisticRegression",
    "DecisionTreeClassifier"
]


class MetaEvaluator():
    def __init__(self, dataset_name: str, window_size: int = 30):
        self.window_size = window_size
        self.dataset_name = dataset_name

    def _get_mean_mse(self, cols: list, data_frame: pd.DataFrame, metric: str=None):
        result_mse = pd.DataFrame(columns=cols).astype(float)
       
        if not metric:
            metric = cols[0].split("_")[0]

        iter_range = range(0, data_frame.shape[0] - self.window_size, self.window_size)

        for result_idx, original_idx in enumerate(iter_range):
            batch = data_frame.iloc[original_idx:original_idx + self.window_size]
            for col in cols:
                result_mse.loc[result_idx, col] = mean_squared_error(batch[metric], batch[col])

        return result_mse.mean(axis=1)

    def _get_result_df(self, filename: str):
        notebook_dir = os.path.dirname(os.path.abspath('.'))  
        correct_path = os.path.join(notebook_dir,filename)

        if not os.path.exists(correct_path):
            print(f"ERRO: Arquivo não encontrado em {correct_path}")
    
        df = pd.read_csv(correct_path).dropna()
        print("correct_path", correct_path)

        metrics = list(set(df.columns).intersection(set(["auc","f1-score","recall", "precision", "kappa"])))
        results = pd.DataFrame()
        for metric in metrics:
            metric_cols = [col for col in df.columns if metric in col]
            with_drift_cols = [col for col in metric_cols if "with_drift" in col]
            without_drift_cols = [col for col in metric_cols if "without_drift" in col]

            # print("with_drift_cols: ", with_drift_cols)
            # print("without_drift_cols", without_drift_cols)
            results[f"{metric}_mse_with_drift"] = self._get_mean_mse(with_drift_cols, df)
            results[f"{metric}_mse_without_drift"] = self._get_mean_mse(without_drift_cols, df)
            results[f"{metric}_mse_baseline"] = self._get_mean_mse([f"last_{metric}"], df, metric)
        return results, metrics

    def _plot_subplot(self, results_df: pd.DataFrame, color: str=COLORS[0], metric="kappa"):
        mtl_with_drift_error = results_df[f"{metric}_mse_with_drift"]
        mtl_without_drift_error = results_df[f"{metric}_mse_without_drift"]
        mtl_with_drift_gain = mtl_without_drift_error - mtl_with_drift_error

        y = mtl_with_drift_gain.cumsum()
        x = np.arange(len(y))
        plt.fill_between(x, y, alpha=0.1, color=color)
        plt.plot(x, y, label=metric, color=color)
        plt.legend(loc=2, fontsize='large')

    def fit(self):
        self.results = {}
        self.metrics = {}
        for base_model in BASE_MODELS:
            filename = f"results/results_dataframes/base_model: {base_model} - dataset: {self.dataset_name}.csv"
            self.results[base_model], self.metrics[base_model] = self._get_result_df(filename)
        
        return self

    def plot_gain(self):
        plt.figure(figsize=(25, 15))
        plt.ylim(bottom=0, top=0.01)  
        plt.suptitle(f"dataset: {self.dataset_name}", fontsize=25)
        show = True
        for base_model_idx, base_model in enumerate(BASE_MODELS):
            plt.subplot(2, 2, base_model_idx + 1)
            for metric_idx, metric in enumerate(self.metrics[base_model]):
                plt.title(f"base_model: {base_model}", fontsize=20)
                self._plot_subplot(self.results[base_model], metric=metric, color=COLORS[metric_idx])


    def _plot_comp_subplot(self, results_df: pd.DataFrame, color: str=COLORS[0], 
                        metric: str="kappa", plot_col: str="proposed_mtl"):
        
        if plot_col == "proposed_mtl":
            # Proposed MTL: baseline vs with_drift
            regressor_error = results_df[f"{metric}_mse_with_drift"]
            baseline_error = results_df[f"{metric}_mse_baseline"]
            gain = baseline_error - regressor_error
            
        elif plot_col == "original_mtl":
            # Original MTL: baseline vs without_drift  
            regressor_error = results_df[f"{metric}_mse_without_drift"]
            baseline_error = results_df[f"{metric}_mse_baseline"]
            gain = baseline_error - regressor_error
            
        elif plot_col == "ideal_regressor":
            # Ideal: ganho máximo possível (erro = 0)
            baseline_error = results_df[f"{metric}_mse_baseline"]
            gain = baseline_error  # Pois ideal teria erro 0
        
        y = gain.cumsum()
        x = np.arange(len(y))
        plt.fill_between(x, y, alpha=0.1, color=color)
        plt.plot(x, y, label=plot_col, color=color)
        plt.legend(loc=2, fontsize='large')

    def plot_original_vs_proposed_mtl_gain(self, metric="kappa", plot_ideal_regressor=True, subplot_index=1):
        for base_model in BASE_MODELS:
            plt.subplot(4, 4, subplot_index)
            if plot_ideal_regressor:
                self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[2], plot_col="ideal_regressor")
            self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[0], plot_col="proposed_mtl")
            self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[1], plot_col="original_mtl")
            subplot_index += 1
            plt.title(f"{metric} - {base_model}")