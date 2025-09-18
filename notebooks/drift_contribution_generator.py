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


# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')


class DriftContributionGenerator():
    def __init__(
        self,
        base_model: str,
        dataset_name: str,
        train_batch_size: str = 200,
        n_models: int = 3,
        select_k_features=1, 
        custom_dir =None
    ):
        self.train_batch_size = train_batch_size
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.n_models = n_models
        self.select_k_features = select_k_features
        self.custom_dir=custom_dir

    def _load_metabase(self) -> None:
        filename = f"basemodel: {self.base_model}  - dataset: {self.dataset_name}"
        self.metabase = pd.read_csv(f"metabase/{self.custom_dir}/{filename} - with_drift_metrics.csv")

    def _create_results_df(self) -> None:
        self.metrics = list(set(self.metabase.columns).intersection(["auc", "kappa", "f1-score", "precision", "recall"]))
        final_cols = self.metrics + [f"last_{metric}" for metric in self.metrics]
        self.results = self.metabase[final_cols]
        for metric in self.metrics:
            for i in range(self.n_models):
                self.results[f"{metric}_pred_{i}_with_drift"] = 0
                self.results[f"{metric}_pred_{i}_without_drift"] = 0

    def _create_meta_models(self):
        self.meta_models = {}
        for metric in self.metrics:
            self.meta_models[metric] = {
                "with_drift": [MetaModel(random_state=i, select_k_features=self.select_k_features) for i in range(self.n_models)],
                "without_drift": [MetaModel(random_state=i) for i in range(self.n_models)],
                }

    def _imp_dict(self, meta_model):
        model = meta_model.model
        # print(f"essa eh o model {type(meta_model)} {meta_model}  {type(model)} {model}")
        importances = np.array(model.feature_importances_, dtype=float)
        # print("essa eh o importances:",
        # importances)
        return dict(zip(model.feature_name_, importances))

    def _get_importances(self):
        importances = {}
        # print(f"items: {self.meta_models.items()}")
        for metric, values in self.meta_models.items():
            importances[metric] = {}
            for drift_flag, models in values.items():
                importances[metric][drift_flag] = [self._imp_dict(m) for m in models]
                    # DEBUG: Analisar importância das features de drift
        # for metric in self.metrics:
        #     print(f"\n=== Importância para {metric} ===")
        #     for i in range(self.n_models):
        #         imp_with = importances[metric]["with_drift"][i]
        #         imp_without = importances[metric]["without_drift"][i]
                
        #         # Verificar importância das features de drift
        #         drift_importances = {k: v for k, v in imp_with.items() if k in self.drift_cols}
        #         top_drift_features = sorted(drift_importances.items(), key=lambda x: x[1], reverse=True)[:5]
                
        #         print(f"Model {i} - Top drift features: {top_drift_features}")
    
        return importances

    def _get_drift_cols(self) -> list:
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
        ]

        self.drift_cols = []
        for col in self.metabase.columns:
            if any(ds in col for ds in drift_suffixes):
                self.drift_cols.append(col)

        # print(f"Colunas de drift identificadas ({len(self.drift_cols)}): {self.drift_cols}")
        # print(f"Total de colunas no metabase: {len(self.metabase.columns)}")

    def _train_metamodels(self, batch: pd.DataFrame):
        for metric in self.metrics:
            features = batch.drop(self.metrics, axis=1)
            non_drift_features = features.drop(self.drift_cols, axis=1)
            target = batch[metric]
            for i in range(self.n_models):
                self.meta_models[metric]["with_drift"][i].fit(features, target)
                self.meta_models[metric]["without_drift"][i].fit(non_drift_features, target)

        # for metric in self.metrics:
        #     print(f"Modelos para {metric}:")
        #     print(f"  Com drift: {[id(m) for m in self.meta_models[metric]['with_drift']]}")
        #     print(f"  Sem drift: {[id(m) for m in self.meta_models[metric]['without_drift']]}")

    def _make_prediction(self, batch: pd.DataFrame):
        features = batch.drop(self.metrics, axis=1)
        non_drift_features = features.drop(self.drift_cols, axis=1)

        # print(f"Features com drift: {features.shape}")
        # print(f"Features sem drift: {non_drift_features.shape}")
        # print(f"Colunas removidas: {set(features.columns) - set(non_drift_features.columns)}")
        
        for metric in self.metrics:
            for i in range(self.n_models):
                a1 = self.meta_models[metric]["with_drift"][i].predict(features)
                a2 = self.meta_models[metric]["without_drift"][i].predict(non_drift_features)

                self.results.iloc[
                    batch.index,
                    self.results.columns.get_loc(f"{metric}_pred_{i}_with_drift")] = \
                    a1
                self.results.iloc[
                    batch.index,
                    self.results.columns.get_loc(f"{metric}_pred_{i}_without_drift")] = \
                    a2
                
                if np.array_equal(a1, a2):
                    print(f"⚠️  AVISO: Previsões idênticas para {metric}_model_{i}")
                # else:
                #     print(f"✅ Previsões diferentes para {metric}_model_{i}")

    def _run_mtl(self):
        for index in range(0, self.metabase.shape[0] - self.train_batch_size, self.train_batch_size):
            train_batch = self.metabase.iloc[index:index + self.train_batch_size]
            self._train_metamodels(train_batch)

            pred_batch = self.metabase.iloc[index + self.train_batch_size:index + 2*self.train_batch_size]
            self._make_prediction(pred_batch)

    def _save_results(self):
        filename = f"base_model: {self.base_model} - dataset: {self.dataset_name} - select_k_features: {int((100*self.select_k_features))}"
        output_dir = Path(f"results/{self.custom_dir}/results_dataframes")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(f"{output_dir}/{filename}.csv", index=False)
        output_dir = Path(f"results/{self.custom_dir}/results_importances")
        output_dir.mkdir(parents=True, exist_ok=True)
        importances = self._get_importances()
        # print(f"Importances:{importances}")
        with open(f"{output_dir}/{filename}.json", "w") as fp:
            json.dump(importances, fp)

    def run(self):
        self._load_metabase()
        self._create_results_df()
        self._create_meta_models()
        self._get_drift_cols()
        self._run_mtl()
        self._save_results()


models = ["RandomForestClassifier", "DecisionTreeClassifier", "LogisticRegression", "SVC"]
datasets  = ["electricity", "powersupply"]
custom_dirs = ["fernanda_weak","henrique_weak"]

if __name__ == "__main__":
    start = time.time()
    print("Estou rodando")
    for dir in custom_dirs[:1]:
        for base_model in models:   
            for dataset_name in datasets[:1]:
                for n_features in range(5, 101, 5):
                    print(f"dir: {dir}, base_model: {base_model} - dataset_name: {dataset_name} - n_features:{n_features}") 
                    d_gen = DriftContributionGenerator(
                        base_model=base_model,
                        dataset_name=dataset_name,
                        train_batch_size=97,
                        select_k_features=(n_features/100),
                        custom_dir=dir
                    )
                    d_gen.run()
    print(f"Finished - elapsed time: {time.time() - start}")