import sys
import os
sys.path.append(os.path.abspath("..")) 
import pandas as pd
import numpy as np
from tqdm import tqdm
from models.meta_learner import MetaLearner
from data.data_loader import DataLoader
import matplotlib.pyplot as plt
import lightgbm as ltb
from data.utils.eda import EDA
import pickle
import warnings
from utils import *
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="The reported value is ignored because this `step` .* is already reported")
warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful")
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

custom_dir = "new_version"

include_drift = [True,False]
for ds_name, dataset in DATASETS_METADATA.items():
    #Main loop iterating through each dataset defined in the metadata.
    print(f"ds_name: {ds_name}")
    
    ETA = dataset["eta"]
    STEP = dataset["step"]
    BASE_TRAIN_SIZE = dataset["base_train_size"]
    TARGET_DELAY = dataset["target_delay"]
    OFFLINE_PHASE_SIZE = dataset["offline_phase_size"]

    for base_model in base_models:
        for drift_conf in include_drift:
            base_model_name = base_model.__name__

            df =  DataLoader.load_data(f"real/{ds_name}.arff")

            FILE_NAME = f"basemodel: {base_model_name}  - dataset: {ds_name}"

            if drift_conf:
                FILE_NAME+=" - with_drift"

            print(f"Rodando para {FILE_NAME}")
            base_model_params = {"verbose": True, "basis_model": base_model, "hyperparameters": hyperparams[base_model_name]}
            meta_learner = MetaLearner( 
                base_model_params=base_model_params,
                meta_model_params={},
                performance_metrics=performance_metrics,
                eta=ETA,
                step=STEP,
                has_dft_mfes=drift_conf,
                target_delay=TARGET_DELAY,
                pca_n_components=None,
                evaluator_avg= "weighted"
            )

            offline_df = df.iloc[:OFFLINE_PHASE_SIZE]
            online_df = df.iloc[OFFLINE_PHASE_SIZE:300]
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
            for c in performance_metrics:
                y_true = mb[c]
                y_pred = mb[f'last_{c}']
                x = range(len(y_true))

                fig = plt.figure(figsize=(25, 5))
                plt.plot(x, y_true, label="original")
                plt.plot(x, y_pred, label="baseline")
                plt.legend(loc="upper left")
        
            meta_learner.save_results(f"{dir}/{FILE_NAME}")
            