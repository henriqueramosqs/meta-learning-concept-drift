import pandas as pd
import numpy as pd
from tqdm import tqdm
from models.meta_learner import MetaLearner
from data.data_loader import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as ltb


performance_metric = "recall"
base_model = RandomForestClassifier()
df =  DataLoader.load_data("real/electricity.arff")
OFFLINE_PHASE_SIZE = 5000
BASE_TRAIN_SIZE = 2000
ETA = 200  
STEP = 30 
TARGET_DELAY = 500

meta_learner = MetaLearner(
    base_model=base_model,
    performance_metrics=[performance_metric],
    has_dft_mfes=False,
    eta=ETA,
    step=STEP,
    target_delay=TARGET_DELAY,
    pca_n_components=None
)

offline_df = df.iloc[:OFFLINE_PHASE_SIZE]
online_df = df.iloc[OFFLINE_PHASE_SIZE:]
online_features = online_df.drop("class")
online_targets = online_df["class"]
meta_learner.fit(offline_df,BASE_TRAIN_SIZE)



with tqdm(total=TARGET_DELAY) as pbar:
    for i, row in online_features.iloc[:TARGET_DELAY].iterrows():
        meta_learner.update(row)
        pbar.update(1)

df = online_features.iloc[TARGET_DELAY:-TARGET_DELAY]

with tqdm(total=df.shape[0]) as pbar:
    for i, row in df.iterrows():
        meta_learner.update(row)
        meta_learner.update_target(online_targets.iloc[i - TARGET_DELAY])
        pbar.update(1)

with tqdm(total=TARGET_DELAY) as pbar:
    for target in online_targets.tail(TARGET_DELAY):
        meta_learner.update_target(target)
        pbar.update(1)