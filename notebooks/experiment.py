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
import pickle

# Defines the performance metrics that the meta-learner will predict.
performance_metric = ["recall","precision","kappa","f1-score"]

# A list of base classification models to be evaluated in the experiments.
base_models = [
        RandomForestClassifier,
        LogisticRegression,
        SVC,
        DecisionTreeClassifier,
    ]

# A dictionary containing specific hyperparameters for each base model.
hyperparams ={
    "RandomForestClassifier": {"max_depth": 6} ,
    "DecisionTreeClassifier": {"max_depth": 6} ,
    "LogisticRegression" :{},
    "SVC": {"probability": True}
}

# A dictionary holding metadata and specific parameters for each dataset to be used in the experiments.
DATASETS_METADATA = {
    "electricity": {
        "dataset_name": "electricity",
        "class_col": "class",
        "base_model_type": "binary_classification",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
    "powersupply": {
        "dataset_name": "powersupply",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
    "airlines": {
        "dataset_name": "airlines",
        "class_col": "Delay",
        "base_model_type": "binary_classification",
        "offline_phase_size": 50000,
        "base_train_size": 20000,
        "eta": 1000,
        "step": 300,
        "target_delay": 2000,
    },
    
    "rialto": {
        "dataset_name": "rialto",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 5000,
        "base_train_size": 2000,
        "eta": 100,
        "step": 30,
        "target_delay": 500,
    },
}

include_dft = [True]

custom_dir = "fernanda_st"

for ds_name, dataset in DATASETS_METADATA.items():
    #Main loop iterating through each dataset defined in the metadata.
    print(f"ds_name: {ds_name}")
    
    ETA = dataset["eta"]
    STEP = dataset["step"]
    BASE_TRAIN_SIZE = dataset["base_train_size"]
    TARGET_DELAY = dataset["target_delay"]
    OFFLINE_PHASE_SIZE = dataset["offline_phase_size"]

    for base_model in base_models:

        base_model_name = base_model.__name__
        for has_dft in include_dft:
            
            df =  DataLoader.load_data(f"real/{ds_name}.arff")

            FILE_NAME = f"basemodel: {base_model_name}  - dataset: {ds_name}"
            if has_dft:
                FILE_NAME += " - with_drift_metrics"
            FILE_NAME

            print(f"Rodando para {FILE_NAME}")
            base_model_params = {"verbose": True, "basis_model": base_model, "hyperparameters": hyperparams[base_model_name]}
            meta_learner = MetaLearner( 
                base_model_params=base_model_params,
                meta_model_params={},
                performance_metrics=performance_metric,
                has_dft_mfes=has_dft,
                eta=ETA,
                step=STEP,
                target_delay=TARGET_DELAY,
                pca_n_components=None,
                evaluator_avg= ("micro" if dataset["base_model_type"]   =="multiclass" else "binary")
            )

            offline_df = df.iloc[:OFFLINE_PHASE_SIZE]
            online_df = df.iloc[OFFLINE_PHASE_SIZE:]
            online_features = online_df.drop("class",axis=1).reset_index(drop=True)
            online_targets = online_df["class"]
            meta_learner.fit(offline_df,BASE_TRAIN_SIZE)

            # Offline phase
            with tqdm(total=TARGET_DELAY) as pbar:
                for i, row in online_features.iloc[:TARGET_DELAY].iterrows():
                    row = pd.DataFrame([row], columns=row.index)
                    meta_learner.update(row)
                    pbar.update(1)

            df = online_features.iloc[TARGET_DELAY:-TARGET_DELAY]

            # Online phase
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

            # Saves results 
            os.makedirs(f"metabase/{custom_dir}", exist_ok=True)
            os.makedirs(f"trained_models/{custom_dir}", exist_ok=True)

            
            mb.to_csv(f"metabase/{custom_dir}/{FILE_NAME}.csv", index=False)
        
            with open(f"trained_models/{custom_dir}/{FILE_NAME}.pickle", "wb") as handle:
                pickle.dump(meta_learner.meta_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
