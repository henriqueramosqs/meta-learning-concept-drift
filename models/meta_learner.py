from collections import defaultdict
import time
import os 
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as ltb
from .mfes_extractors  import PsiCalculator, Udetector, DomainClassifier, OmvPht
from .mfes_extractors  import ADWINDetector, KSWINDetector, HDDMADetector, HDDMWDetector,KSWINDetector,DDMDetector
from .mfes_extractors import StatsMFesExtractor, DBSCANMfesExtractor, SqsiCalculator,KmeansMfesExtractor
from .mfes_extractors import *
from eval import Evaluator
from data.data_loader import DataLoader
from data.utils.eda import EDA
from .meta_data_manager import MetaDataManager
from .base_data_manager import BaseDataManager
from .meta_model import MetaModel
from .base_model import BaseModel
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score


# Defines the valid theoretical range for each performance metric.
# Used to clip the predictions of the meta-models.
metrics_range ={
    "precision": (0, 1),
    "recall": (0, 1),
    "f1-score": (0, 1),
    "kappa": (-1, 1),
}

class MetaLearner():
    """
    Orchestrates the meta learning experiment, integrating BaseDataManager,
    MetaDataManager, MetaModel, BaseModel, etc
    """
    def __init__(
            self,
            base_model_params,
            meta_model_params,
            performance_metrics:list,
            has_dft_mfes:bool,
            eta:int,
            step:int,
            target_delay:int,
            pca_n_components:int,
            evaluator_avg=None,
            eval_time_mode:bool = False
        ):
        self.base_model = BaseModel(**base_model_params)
        self.performance_metrics =performance_metrics
        self.has_dft_mfes = has_dft_mfes
        self.eta = eta
        self.step = step
        self.mfes_extractors = []
        self.metabase = MetaDataManager(pca_n_components=pca_n_components,target_cols=self.performance_metrics)
        self.basedata = BaseDataManager(batch_size=eta,step=step)
        self.elapsed_time = defaultdict(int) 
        self.evaluator = Evaluator(evaluator_avg)
        self.target_delay = target_delay
        self.eval_time_mode = eval_time_mode
        self.meta_models = {metric: MetaModel() for metric in self.performance_metrics}

    def _limit_metric_value(self, value: float, metric_name: str) -> float:
        """
        Clips a predicted metric value to its valid theoretical range.

        Args:
            value (float): The predicted performance metric value.
            metric_name (str): The name of the metric to look up its range.

        Returns:
            float: The clipped value.
        """

        value = max(value,metrics_range[metric_name][0])
        value = min(value, metrics_range[metric_name][1])
        return value

    def _train_base_models(self, df: pd.DataFrame) -> None:
        """
        Trains the base model on an initial dataset.

        Args:
            df (pd.DataFrame): The training data for the base model. Must contain a 'class' column.
        """
        features = df.drop("class", axis=1)
        target = df["class"]
        self.base_model.fit(features,target)

    def _fit_mfes(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Initializes and fits the meta-feature (MFEs) extractors on a reference dataset.        
        Args:
            df (pd.DataFrame): The reference data for fitting the extractors.
        """
        features = df.rename(columns={"class":"prediction"})
        feature_cols = features.columns
        pred_proba = self.base_model.predict_proba(features.drop("prediction",axis=1))
        score_cols = []
        for idx, pred in enumerate(pred_proba.T):
            features[f"predict_proba_{idx}"] = pred
            score_cols.append(f"predict_proba_{idx}")
 
        self.mfes_extractors = [
            StatsMFesExtractor().fit(),
            DBSCANMfesExtractor().fit(),
            KmeansMfesExtractor().fit()
        ]
        if self.has_dft_mfes:
            self.mfes_extractors += [
                PsiCalculator().fit(features),
                DomainClassifier().fit(features),
                OmvPht(score_cols=score_cols).fit(features),
                SqsiCalculator(score_cols=score_cols).fit(features),
                Udetector(prediction_col="prediction").fit(features),
                KSWINDetector(feature_cols).fit(features),
                BhattacharyyaDetector(feature_cols).fit(features),
                HellingerDistanceDetector(feature_cols).fit(features),
                JensenShanonDetector(feature_cols).fit(features),
                EMDDetector(feature_cols).fit(features),
                EnergyDistanceDetector(feature_cols).fit(features),
            ]
        

    def _get_baseline(self) -> dict:
        """
        Retrieves the performance metrics from the most recent labeled batch.

        Returns:
            dict: A dictionary where keys are like 'last_precision', 'last_recall', etc.
        """

        batch = self.metabase.get_last_tageted_row()[self.performance_metrics]
        res =  {f"last_{metric}": value for metric, value in batch.to_dict().items()}
        return res

    def _get_mfes(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Extracts all meta-features for a given data batch in parallel.

        Args:
            df (pd.DataFrame): The data batch to extract features from.

        Returns:
            pd.DataFrame: A single-row DataFrame containing all extracted meta-features.
        """
        mf_dict= {}

        if not self.eval_time_mode:
            with ThreadPoolExecutor() as executor:
                futures = []
                for extractor in self.mfes_extractors:
                    futures.append(executor.submit(self._extract_metric, extractor, df))
                
                for future in futures:
                    metric_name, result, elapsed = future.result()
                    mf_dict.update(result)
                    self.elapsed_time[metric_name] += elapsed
        else:
            for extractor in self.mfes_extractors:
                metric_name, result, elapsed = self._extract_metric(extractor,df)
                mf_dict.update(result)
                self.elapsed_time[metric_name] += elapsed
            
        return pd.DataFrame([mf_dict])

    def _get_meta_labels(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Calculates the true performance metrics (labels) for a labeled data batch.
        
        Args:
            df (pd.DataFrame): The labeled data batch, containing both true labels ('class')
                               and predictions ('prediction').

        Returns:
            dict: A dictionary of calculated performance metrics.
        """
        y_true = df["class"]
        y_pred = df["prediction"]

        metrics = {
            metric: self.evaluator.evaluate(metric, y_true,y_pred) for metric in self.performance_metrics
        }

        return metrics

    def _get_train_metabase(self, target_col:str=None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Retrieves the training data for the meta-models from the MetaDataManager.

        Args:
            target_col (str, optional): The specific performance metric to be used as the target. 
                                        If None, only features are returned. Defaults to None.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing the meta-features and the target series.
        """
        meta_base = self.metabase.get_train_batch()

        features = meta_base.drop([col for col in self.performance_metrics if col in meta_base.columns], axis=1)

        target= None
        if(target_col!=None):
            target = meta_base[target_col]
        return  features, target

    def _extract_metric(self, extractor, df:pd.DataFrame) -> tuple:
        """
        A helper function to run a single extractor and time its execution.

        Args:
            extractor: An instance of a meta-feature extractor.
            df (pd.DataFrame): The data batch to process.

        Returns:
            tuple: A tuple containing the extractor's name, its result (a dict of features),
                   and the time it took to run.
        """
        start = time.time()
        result = extractor.evaluate(df)
        elapsed = time.time() - start
        return (extractor.__class__.__name__, result, elapsed)
        
    def _get_last_performances(self, meta_base: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lagged performance features. This shifts the performance metrics from previous
        time steps to be used as input features for the current step.

        Args:
            meta_base (pd.DataFrame): The meta-dataset.

        Returns:
            pd.DataFrame: The meta-dataset with added 'last_<metric>' columns.
        """
        start = time.time()
        for metric in self.performance_metrics:
            col_name = f"last_{metric}"
            meta_base.loc[:, col_name] = meta_base[metric].shift(self.target_delay)
        elapsed = time.time() - start
        self.elapsed_time["ScoringMetrics"] += elapsed
        return meta_base
    
    def _init_base_data(self,df:pd.DataFrame)->None:
        """
        Initializes the BaseDataManager with the initial stream of data after the base model
        has been trained. It adds base model predictions and probabilities to the dataframe.

        Args:
            df (pd.DataFrame): The initial data stream for meta-learning.
        """
        features = df.drop("class",axis=1)

        pred_proba = self.base_model.predict_proba(features)
        df = df.assign(**{f"predict_proba_{idx}": pred for idx, pred in enumerate(pred_proba.T)})
        
        df["prediction"] = self.base_model.predict(features)

        self.basedata.set_init_df(df)

    
    def _train_meta_model(self,is_first=False) -> None:
        """
        Trains each meta-model on the current meta-dataset. A separate model is trained
        for each performance metric.
        """
        for metric in self.performance_metrics:
            features, target = self._get_train_metabase(metric)
            self.meta_models[metric].fit(features, target)

    def _init_metabase(self)->None: 
        """
        Creates the initial meta-dataset from the base data. It iterates through the
        data in batches, extracts meta-features, calculates actual performance (meta-labels),
        and assembles them into a training set for the meta-models.
        """
        df = self.basedata.get_raw()

        batches = [
            df.iloc[i:i + self.eta]
            for i in range(0, df.shape[0]-self.eta, self.step)
        ]

        meta_base = pd.DataFrame()

        for i, batch in enumerate(batches):
            batch_features = batch.drop("class",axis=1)
            mfes_df = self._get_mfes(batch_features)
            meta_labels = self._get_meta_labels(batch)
            meta_labels_df = pd.DataFrame(meta_labels, index=[i])
            meta_batch = pd.concat([mfes_df.reset_index(drop=True), 
                               meta_labels_df.reset_index(drop=True)], axis=1)
            meta_base = pd.concat([meta_base, meta_batch], ignore_index=True,axis=0)
            
        meta_base = self._get_last_performances(meta_base)
        self.metabase.set_init_df(meta_base)


    def update(self, new_instance_df: pd.DataFrame) -> None:
        """
        Processes a new, unlabeled data instance from the stream. If a new batch is
        completed, it extracts meta-features and uses the meta-models to predict future performance.

        Args:
            new_instance_df (pd.DataFrame): A single-row DataFrame with a new instance.
        """


        pred_proba = self.base_model.predict_proba(new_instance_df)
        new_instance_df["prediction"] = self.base_model.predict(new_instance_df)[0]
        new_instance_df = new_instance_df.assign(**{f"predict_proba_{idx}": pred for idx, pred in enumerate(pred_proba.T)})
        
        self.basedata.update(new_instance_df)

        if self.basedata.has_new_batch():
            baseline = self._get_baseline()
            batch = self.basedata.get_last_batch()

            mfes_df = self._get_mfes(batch)

            mfes_df = mfes_df.assign(**baseline)
            
            start = time.time()

            predictions = {
                f"meta_predict_{metric}": model.predict(mfes_df)
                for metric, model in self.meta_models.items()
            }
            for key, value in predictions.items():
                metric = key.replace("meta_predict_", "")
                
                vectorized_limit = np.vectorize(lambda x: self._limit_metric_value(x, metric))
                predictions[key] = vectorized_limit(value)

            elapsed = time.time() - start
            self.elapsed_time["ScoringMetrics"] += elapsed
            mfes_df =  mfes_df.assign(**predictions)
            self.metabase.update(mfes_df)

    def update_target(self, target) -> None:
        """
        Processes a new true label when it becomes available. If a labeled batch is
        completed, it calculates the actual performance and updates the meta-dataset,
        potentially triggering a retraining of the meta-models.

        Args:
            target: The true label for an older instance.
        """
        self.basedata.update_target(target)

        if self.basedata.has_new_targeted_batch():
            batch = self.basedata.get_targeted_batch()

            meta_labels = self._get_meta_labels(batch)
            if self.metabase.cur_batch_size == self.step:
                self._train_meta_model()


    def fit(self,train_df: pd.DataFrame, base_train_size:int)->None:
        """
        Orchestrates the  offline training process.

        Args:
            train_df (pd.DataFrame): The complete training dataset.
            base_train_size (int): The number of instances to use for training the base model.
                                   The rest will be used for training the meta-models.

        Returns:
            self: The fitted MetaLearner instance.
        """
        base_train = train_df[:base_train_size]
        meta_train = train_df[base_train_size:]
        self._train_base_models(base_train)
        self._fit_mfes(base_train.copy())
        self._init_base_data(meta_train.copy())
        self._init_metabase()   
        self._train_meta_model(is_first=True)
        
        features, _ = self._get_train_metabase()
        for metric, model in self.meta_models.items():
            y_pred = model.predict(features)
            self.metabase.set_pred(prediction = y_pred, prediction_col=f"meta_predict_{metric}")
        return self
    

    def save_results(self,dest):
        if not self.eval_time_mode:
            # os.makedirs(f"metabase/{dest}", exist_ok=True)
            # os.makedirs(f"trained_models/{dest}", exist_ok=True)
            
            mb = self.metabase.metabase
                
            mb.to_csv(f"metabase/{dest}.csv", index=False)
       
            with open(f"trained_models/{dest}.pickle", "wb") as handle:
                pickle.dump(self.meta_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
           os.makedirs(f"time_elapsed", exist_ok=True) 
           df = pd.DataFrame(
                           {key:[value] for key,value in self.elapsed_time.items()}
                             )
           df.to_csv(f"time_elapsed/{dest}")
        
if __name__ == "__main__":
    base_model = RandomForestClassifier()
    performance_metrics =["precision","recall", "f1-score","kappa"]
    df =  DataLoader.load_data("real/electricity.arff")
    meta_learner = MetaLearner(base_model=base_model,performance_metrics=performance_metrics,
                            has_dft_mfes=True,eta=100,step=20,target_delay=500, pca_n_components=5)
    meta_learner.fit(df,300)
    meta_learner.update(df.drop("class",axis=1).iloc[301])
