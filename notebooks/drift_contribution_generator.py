import sys

sys.path.insert(0,'..')
sys.path.insert(0,'../..')

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from models import MetaModel
from eval.evaluator import Evaluator
from data.utils.eda import EDA 
from utils import *

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


class DriftContributionGenerator():
    """
    Analyzes the contribution of drift-related meta-features 
    in predicting the performance of a base model. It works by training two sets of 
    meta-models in parallel: one that uses all available meta-features (including 
    drift metrics) and another that uses only non-drift-related meta-features. 
    By comparing the predictions and feature importances of these two sets of models, 
    it quantifies the value of drift detection features.
    """
    def __init__(
        self,
        base_model: str,
        dataset_name: str,
        train_batch_size: str = 200,
        n_models: int = 3,
        select_k_features=1, 
        custom_dir =None
    ):
        
        """
        Initializes the experiment with specified configurations.

        Args:
            base_model (str): The name of the base model being evaluated (e.g., 'RandomForestClassifier').
            dataset_name (str): The name of the dataset used (e.g., 'airlines').
            train_batch_size (int, optional): The number of instances in each batch used for training the meta-models.
            n_models (int, optional): The number of meta-models to train for each scenario (with/without drift) 
                                      to create an ensemble or average results.
            select_k_features (float, optional): The percentage of top features to select for the models trained 
                                                 'with_drift'. A value of 1 means all features are used.
            custom_dir (str, optional): A subdirectory name inside results/ for organizing input and output files.
        """
        
        self.train_batch_size = train_batch_size
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.n_models = n_models
        self.select_k_features = select_k_features
        self.custom_dir=custom_dir
        self.show = True
      

    def _load_metabase(self) -> None:
        """
        Loads the pre-generated meta-dataset from a CSV file.
        """
        filename = f"basemodel: {self.base_model}  - dataset: {self.dataset_name}"
        self.metabase_drift = pd.read_csv(f"metabase/{self.custom_dir}/{filename} - with_drift_metrics.csv")
        self.metabase_no_drift = pd.read_csv(f"metabase/{self.custom_dir}/{filename}.csv")

        self.metabase_drift.rename(columns=lambda col: col.replace('bhattach aryya', 'bhattacharyya'), inplace=True)
        self.metabase_drift = self.metabase_drift.drop(columns=["original_idx", "data_type"], errors='ignore')
        self.metabase_no_drift = self.metabase_no_drift.drop(columns=["original_idx", "data_type"], errors='ignore')

        self.drift_to_drop  = []
        self.no_drift_to_drop = []
        print(filename,flush=True)

        # self.drift_to_drop  = [col for col in self.metabase_drift.columns if ("predict" in col or "last" in col)]
        # self.no_drift_to_drop = [col for col in self.metabase_no_drift.columns if ("predict" in col or "last" in col)]

    def _create_results_df(self) -> None:
        """
        Initializes a DataFrame to store the results of the experiment.
        It includes the actual performance metrics and creates columns
        to hold the predictions from both sets of meta-models (with and without drift features).
        """
        self.metrics = list(set(self.metabase_drift.columns).intersection(["auc", "kappa", "f1-score", "precision", "recall"]))
        final_cols = self.metrics + [f"last_{metric}" for metric in self.metrics]
        self.results = self.metabase_drift[final_cols]
        for metric in self.metrics:
            for i in range(self.n_models):
                self.results[f"{metric}_pred_{i}_with_drift"] = 0
                self.results[f"{metric}_pred_{i}_without_drift"] = 0

    def _create_meta_models(self):
        """
        Instantiates the meta-models for the experiment. For each performance metric,
        it creates two sets of models: one to be trained with drift features and one without.
        """

        self.meta_models = {}
        for metric in self.metrics:
            self.meta_models[metric] = {
                "with_drift": [MetaModel(random_state=i, select_k_features=self.select_k_features) for i in range(self.n_models)],
                "without_drift": [MetaModel(random_state=i) for i in range(self.n_models)],
                }

    def _imp_dict(self, meta_model):
        """
        A helper function to extract feature importances from a trained meta-model.
        Args:
            meta_model (MetaModel): A trained MetaModel instance.

        Returns:
            dict: A dictionary mapping feature names to their importance scores.
        """
        model = meta_model.model
        importances = np.array(model.feature_importances_, dtype=float)
        return dict(zip(model.feature_name_, importances))

    def _get_importances(self):
        """
        Aggregates feature importances from all trained meta-models across all metrics
        and scenarios (with/without drift).
        
        Returns:
            dict: A nested dictionary containing the feature importances.
        """
        importances = {}
        for metric, values in self.meta_models.items():
            importances[metric] = {}
            for drift_flag, models in values.items():
                importances[metric][drift_flag] = [self._imp_dict(m) for m in models]

        return importances

    def _get_drift_cols(self) -> list:
        """
        Identifies columns in the metabase that are considered drift-related features
        based on a predefined list of suffixes and substrings.
        """
        drift_suffixes = [
            "adwin_",
            "dbscan_",
            "ddm_",
            "dc_",
            "hddma_",
            "hddmw_",
            "kmeans_",
            "kswin_",
            "omv_pth_",
            "psi_",
            "u_detect_",
            "overlap_",
            "_ks_statistic",
            "_ks_pvalue",
            "sqsi_drift_flag",
            "predict",
            "last",
            "bhattacharyya",
            "energy_distance_",
            "emd_",
            "hellinger_",
            "jensen_shanon_",
        ]

        self.drift_cols = []
        for col in self.metabase.columns:
            if any(ds in col for ds in drift_suffixes):
                self.drift_cols.append(col)

        # print(f"Colunas de drift identificadas ({len(self.drift_cols)}): {self.drift_cols}")
        # print(f"Total de colunas no metabase: {len(self.metabase.columns)}")

    def _train_metamodels(self, drift_batch: pd.DataFrame,no_drift_bach:pd.DataFrame):
        """
        Trains all meta-models on a given batch of data. For each performance metric,
        it trains models with and without the drift-related features.

        Args:
            batch (pd.DataFrame): The batch of data from the metabase to use for training.
        """
        for metric in self.metrics:
            features_drift = drift_batch.drop(self.metrics+self.drift_to_drop, axis=1)
            features_no_drift = no_drift_bach.drop(self.metrics+self.no_drift_to_drop, axis=1)
            # print(f"features: { [col for col in features.columns.to_list() if ("bhattach" in col) ]}")
            # print(f"non drift features: {non_drift_features.columns.to_list()}")
            target = drift_batch[metric]

            for i in range(self.n_models):
                self.meta_models[metric]["with_drift"][i].fit(features_drift, target)
                self.meta_models[metric]["without_drift"][i].fit(features_no_drift, target)

    def _make_prediction(self, drift_batch: pd.DataFrame,no_drift_bach:pd.DataFrame):
        """
        Makes predictions using the trained meta-models on a new batch of data.

        Args:
            batch (pd.DataFrame): The batch of data to make predictions on.
        """
        if self.show:
            print("s1: ",len(self.metrics+ self.drift_to_drop))
            print("s2: ",len(self.metrics+ self.no_drift_to_drop))
            self.show=False

        features_drift = drift_batch.drop(self.metrics+ self.drift_to_drop, axis=1)
        features_no_drift = no_drift_bach.drop(self.metrics+self.no_drift_to_drop, axis=1)

        # print(f"Features com drift: {features.shape}")
        # print(f"Features sem drift: {non_drift_features.shape}")
        # print(f"Colunas removidas: {set(features.columns) - set(non_drift_features.columns)}")
        
        for metric in self.metrics:
            for i in range(self.n_models):
                a1 = self.meta_models[metric]["with_drift"][i].predict(features_drift)
                a2 = self.meta_models[metric]["without_drift"][i].predict(features_no_drift)

                self.results.iloc[
                    drift_batch.index,
                    self.results.columns.get_loc(f"{metric}_pred_{i}_with_drift")] = \
                    a1
                self.results.iloc[
                    drift_batch.index,
                    self.results.columns.get_loc(f"{metric}_pred_{i}_without_drift")] = \
                    a2
                
                if np.array_equal(a1, a2):
                    print(f"⚠️  AVISO: Previsões idênticas para {metric}_model_{i}")
                # else:
                #     print(f"✅ Previsões diferentes para {metric}_model_{i}")

    def _run_mtl(self):
        """
        Executes the main meta-learning loop, simulating a streaming environment.
        It iterates through the metabase in non-overlapping windows, using one window
        for training and the subsequent one for prediction (interleaved test-then-train).
        """
        for index in range(0, self.metabase_drift.shape[0] - self.train_batch_size, self.train_batch_size):
            train_batch_drift = self.metabase_drift.iloc[index:index + self.train_batch_size]
            train_batch_no_drift = self.metabase_no_drift.iloc[index:index + self.train_batch_size]
            self._train_metamodels(train_batch_drift,train_batch_no_drift)

            pred_batch_drift = self.metabase_drift.iloc[index + self.train_batch_size:index + 2*self.train_batch_size]
            pred_batch_no_drift = self.metabase_no_drift.iloc[index + self.train_batch_size:index + 2*self.train_batch_size]
            self._make_prediction(pred_batch_drift,pred_batch_no_drift)

    def _save_results(self):
        """
        Saves the final results of the experiment, including the predictions DataFrame
        and the feature importances dictionary, to CSV and JSON files, respectively.
        """
        filename = f"base_model: {self.base_model} - dataset: {self.dataset_name} - select_k_features: {int((100*self.select_k_features))}"
        output_dir = Path(f"results/{self.custom_dir}_corrected_windows_no_leak_no_pred/results_dataframes")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(f"{output_dir}/{filename}.csv", index=False)
        output_dir = Path(f"results/{self.custom_dir}_corrected_windows_no_leak_no_pred/results_importances")
        output_dir.mkdir(parents=True, exist_ok=True)
        importances = self._get_importances()
        # print(f"Importances:{importances}")
        with open(f"{output_dir}/{filename}.json", "w") as fp:
            json.dump(importances, fp)

    def run(self):
        """
        Orchestrates the entire workflow of the drift contributinon analyzes 
        """
        self._load_metabase()
        self._create_results_df()
        self._create_meta_models()
        # self._get_drift_cols()
        self._run_mtl()
        self._save_results()

custom_dirs = ["henrique_st"]

def get_window_size(metadata: dict) -> int:
    mtl_size = metadata["offline_phase_size"] - metadata["base_train_size"]
    eta = metadata["eta"]
    step = metadata["step"]
    window_size = (mtl_size - eta)/step
    return int(np.ceil(window_size))  

if __name__ == "__main__":
    start = time.time()
    print("Estou rodando")
    for dir in custom_dirs:
        for  dataset_name, metadata in DATASETS_METADATA.items():
            if(dataset_name=="powersupply" | dataset_name=="electricity" ):
                continue
            for base_model in base_models:   
                base_model_name = base_model.__name__
                for n_features in range(100, 101, 5):
                    print(f"dir: {dir}, base_model: {base_model_name} - dataset_name: {dataset_name} - n_features:{n_features}") 
                    d_gen = DriftContributionGenerator(
                        base_model=base_model_name,
                        dataset_name=dataset_name,
                        train_batch_size=get_window_size(metadata),
                        select_k_features=(n_features/100),
                        custom_dir=dir
                    )
                    d_gen.run()
    print(f"Finished - elapsed time: {time.time() - start}")
