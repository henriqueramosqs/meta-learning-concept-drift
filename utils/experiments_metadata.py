
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC

performance_metrics = ["recall","kappa","f1-score","precision"]

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

DATASETS_METADATA = {

    "powersupply": {
        "dataset_name": "powersupply",
        "class_col": "class",
        "base_model_type": "multiclass",
        "offline_phase_size": 200,
        "base_train_size": 100,
        "eta": 100,
        "step": 100,
        "target_delay": 100,
    },

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