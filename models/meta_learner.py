from collections import defaultdict
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as ltb
from .mfes_extractors  import PsiCalculator, Udetector, DomainClassifier, OmvPht
from .mfes_extractors import StatsMFesExtractor, DBSCANMfesExtractor, SqsiCalculator,KmeansMfesExtractor
from eval.evaluator import Evaluator
from data.data_loader import DataLoader
from .meta_data_manager import MetaDataManager
from .base_data_manager import BaseDataManager
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

# Given a dataset, performs the meta-learning task 
# python3 -m models.meta_learner
# class MetaLearner:

# Um base model, um meta model para cada performance_metric
class MetaLearner():
    def __init__(self,base_model,performance_metrics:list,has_dft_mfes:bool,eta:int,step:int,target_delay:int):
        self.base_model = base_model
        self.performance_metrics =performance_metrics
        self.has_dft_mfes = has_dft_mfes
        self.eta = eta
        self.step = step
        self.mfes_extractors = []
        self.metabase = MetaDataManager()
        self.basedata = BaseDataManager()
        self.elapsed_time = defaultdict(int) 
        self.evaluator = Evaluator()
        self.target_delay = target_delay
        self.meta_models = {metric: ltb.LGBMRegressor(verbosity=-1) for metric in self.performance_metrics}

    def _train_base_models(self, df: pd.DataFrame) -> None:
        features = df.drop("class", axis=1)
        target = df["class"]
        print("Columns que o base model recebeu no train: ",features.columns)
        self.base_model.fit(features,target)

    def _fit_mfes(self,df:pd.DataFrame)->pd.DataFrame:
        features = df.rename(columns={"class":"prediction"})
        # print("cols: ",features.columns)
        pred_proba = self.base_model.predict_proba(features.drop("prediction",axis=1))
        score_cols = []
        for idx, pred in enumerate(pred_proba.T):
            features[f"predict_proba_{idx}"] = pred
            score_cols.append(f"predict_proba_{idx}")
 
        self.mfes_extractors = [
            StatsMFesExtractor().fit(),
            DBSCANMfesExtractor().fit(),
        ]

        if self.has_dft_mfes:
            self.mfes_extractors += [
                PsiCalculator().fit(features),
                DomainClassifier().fit(features),
                OmvPht(score_cols=score_cols).fit(features),
                SqsiCalculator(score_cols=score_cols).fit(features),
                Udetector(prediction_col="prediction").fit(features),
            ]
        

    def _get_mfes(self,df:pd.DataFrame)->pd.DataFrame:
        mf_dict= {}
    
        with ThreadPoolExecutor() as executor:
            futures = []
            for extractor in self.mfes_extractors:
                futures.append(executor.submit(self._extract_metric, extractor, df))
            
            for future in futures:
                metric_name, result, elapsed = future.result()
                mf_dict.update(result)
                self.elapsed_time[metric_name] += elapsed
        
        return pd.DataFrame([mf_dict])

    def _get_meta_labels(self,df:pd.DataFrame)->pd.DataFrame:
        y_true = df["class"]
        y_pred = df["prediction"]
        metrics = {
            metric: self.evaluator.evaluate(metric, y_true,y_pred) for metric in self.performance_metrics
        }
        return metrics

    def _get_train_metabase(self, target_col:str=None) -> tuple[pd.DataFrame, pd.Series]:
        meta_base = self.metabase.get_train_metabase()
        features = meta_base.drop([col for col in self.performance_metrics if col in meta_base.columns], axis=1)

        print(meta_base.info())
        print(f"Columns na metabase: {meta_base.columns}")
        target= None
        if(target_col!=None):
            target = meta_base[target_col]
        return  features, target

    def _extract_metric(self, extractor, df:pd.DataFrame) -> tuple:
        start = time.time()
        result = extractor.evaluate(df)
        elapsed = time.time() - start
        return (extractor.__class__.__name__, result, elapsed)
        
    def _get_last_performances(self, meta_base: pd.DataFrame) -> pd.DataFrame:
        for metric in self.performance_metrics:
            col_name = f"baseline_{metric}"
            meta_base.loc[:, col_name] = meta_base[metric].shift(self.target_delay)
        return meta_base
    
    def _init_base_data(self,df:pd.DataFrame)->None:
        features = df.drop("class",axis=1)

        pred_proba = self.base_model.predict_proba(features)
        df = df.assign(**{f"predict_proba_{idx}": pred for idx, pred in enumerate(pred_proba.T)})
        
        df["prediction"] = self.base_model.predict(features)
        self.basedata.set_init_df(df)

    
    def _train_meta_model(self) -> None:
        for metric in self.performance_metrics:
            features, target = self._get_train_metabase(metric)
            self.meta_models[metric].fit(features, target)

    def _init_metabase(self)->None: 
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
            meta_base = pd.concat([meta_base, meta_batch], ignore_index=True)
        
        meta_base = self._get_last_performances(meta_base)
        self.metabase.set_init_df(pd.DataFrame(meta_base))


    def update(self, new_instance: pd.DataFrame) -> None:
        new_instance_df = pd.DataFrame(new_instance).T

        pred_proba = self.base_model.predict_proba(new_instance_df)
        print(pred_proba)

        print("Columns que o base model recebeu no update: ",new_instance_df.columns)
        new_instance_df["prediction"] = self.base_model.predict(new_instance_df)[0]
        new_instance_df = new_instance_df.assign(**{f"predict_proba_{idx}": pred for idx, pred in enumerate(pred_proba.T)})
        
        self.basedata.update(new_instance)

        # # If there is a new batch for calculating meta fetures
        if self.basedata.has_new_batch():
        #     baseline = self._get_baseline()
        #     batch = self.baselevel_base.get_batch()
        #     meta_features = self._get_meta_features(batch)
        #     meta_features[list(baseline.keys())] = list(baseline.values())
        #     meta_features[self.metabase.prediction_col] = self.meta_model.predict(meta_features)
        #     self.metabase.update(meta_features)


    def fit(self,train_df: pd.DataFrame, base_train_size:int)->None:
        base_train = train_df[:base_train_size]
        meta_train = train_df[:base_train_size]
        self._train_base_models(base_train)
        self._fit_mfes(base_train.copy())
        self._init_base_data(meta_train.copy())
        self._init_metabase()   
        self._train_meta_model()

        features, _ = self._get_train_metabase()
        print(features.columns,self.meta_models.items())
        for metric, model in self.meta_models.items():
            y_pred = model.predict(features)
            self.metabase.set_pred(prediction = y_pred, prediction_col=f"meta_predict_{metric}")
        return self

    

if __name__ == "__main__":
    base_model = RandomForestClassifier()
    performance_metrics =["precision","recall", "f1-score"]
    df =  DataLoader.load_data("real/electricity.arff")
    meta_learner = MetaLearner(base_model=base_model,performance_metrics=performance_metrics,
                            has_dft_mfes=True,eta=100,step=20,target_delay=500)
    meta_learner.fit(df,300)
    meta_learner.update(df.drop("class",axis=1).iloc[301])
