import sys
import os
sys.path.append(os.path.abspath("..")) 
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.meta_learner import MetaLearner
from data.data_loader import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import lightgbm as ltb
from data.utils.eda import EDA


performance_metric = ["recall","precision","kappa","f1-score"]
base_models = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        LogisticRegression(),
        SVC()
    ]

datasets = [
    "electricity",
    "Rialto",
    "powersupply",
    "airlines"
    ]
include_dft = [True, False]
df =  DataLoader.load_data("real/electricity.arff")
OFFLINE_PHASE_SIZE = 5000
BASE_TRAIN_SIZE = 2000
ETA = 200  
STEP = 30 
TARGET_DELAY = 500


for base_model in base_models:
    for dataset in datasets: 
        for has_dft in include_dft:
            FILE_NAME = f"basemodel: {base_model.__name__}  - dataset: {dataset}"
            if has_dft:
                FILE_NAME += " - with_drift_metrics"
            FILE_NAME
            meta_learner = MetaLearner(
                base_model=base_model,
                performance_metrics=performance_metric,
                has_dft_mfes=True,
                eta=ETA,
                step=STEP,
                target_delay=TARGET_DELAY,
                pca_n_components=None
            )

            offline_df = df.iloc[:OFFLINE_PHASE_SIZE]
            online_df = df.iloc[OFFLINE_PHASE_SIZE:]
            online_features = online_df.drop("class",axis=1).reset_index(drop=True)
            online_targets = online_df["class"]
            meta_learner.fit(offline_df,BASE_TRAIN_SIZE)

            with tqdm(total=TARGET_DELAY) as pbar:
                for i, row in online_features.iloc[:TARGET_DELAY].iterrows():
                    row = pd.DataFrame([row], columns=row.index)
                    meta_learner.update(row)
                    pbar.update(1)

            df = online_features.iloc[TARGET_DELAY:-TARGET_DELAY]


            with tqdm(total=df.shape[0]) as pbar:
                for i, row in df.iterrows():
                    row = pd.DataFrame([row], columns=row.index)
                    meta_learner.update(row)
                    meta_learner.update_target(online_targets.iloc[i - TARGET_DELAY])
                    pbar.update(1)
                

            with tqdm(total=TARGET_DELAY) as pbar:
                for target in online_targets.tail(TARGET_DELAY):
                    meta_learner.update_target(target)
                    pbar.update(1)

            mb = meta_learner.metabase.metabase

            for c in performance_metric:
                y_true = mb[c]
                y_pred = mb[f'last_{c}']
                x = range(len(y_true))

                fig = plt.figure(figsize=(25, 5))
                plt.plot(x, y_true, label="original")
                plt.plot(x, y_pred, label="baseline")
                plt.legend(loc="upper left")

            meta_learner.elapsed_time

            mb.to_csv(f"metabase/{FILE_NAME}.csv", index=False)