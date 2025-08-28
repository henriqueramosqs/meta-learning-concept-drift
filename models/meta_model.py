import numpy as np
from typing import Tuple
import pandas as pd
import lightgbm as ltb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# hyperparam optimization
from optuna.integration import LightGBMPruningCallback
import optuna

# Macros
DEFAULT_N_FOLDS = 5
VERBOSE = True
R_STATE = 2022
DEFAULT_N_TRIALS = 10

def default_param_map(trial):
    """Param map to be used for optuna optimization."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 25, step=1),
        "max_depth": trial.suggest_int("max_depth", 3, 8, step=1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),  # ✅ Adicionado
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),  # ✅ Adicionado
    }

class MetaModel():
    def __init__(
        self,
        param_map = default_param_map,
        n_folds: int = DEFAULT_N_FOLDS,
        verbose: bool = False,
        n_trials: bool = DEFAULT_N_TRIALS,
        random_state: int = R_STATE,
        select_k_features: Tuple[int, float] = None,
        ):
        self.param_map = param_map
        self.n_folds = n_folds
        self.verbose = verbose
        self.n_trials = n_trials
        self.random_state = random_state
        self.select_k_features = select_k_features
        self.best_hyperparams = {}
        self.model = None
        self.feature_list = None
        optuna.logging.set_verbosity(optuna.logging.WARNING)  
        
    def _objective(self, trial, features: pd.DataFrame, target: pd.Series):
        """Time series cross validation for finding the best hyperparam"""
        cross_val = TimeSeriesSplit(n_splits=5)
        cv_scores = np.empty(5)
        hyperparams = self.param_map(trial)
        
        # ✅ DEBUG: Verificar dados de entrada
        # print(f"=== Objective DEBUG ===")
        # print(f"Features shape: {features.shape}")
        # print(f"Target range: [{target.min():.3f}, {target.max():.3f}]")
        # print(f"Target std: {target.std():.3f}")
        # print(f"Target unique values: {len(target.unique())}")
        
        for idx, (train_idx, test_idx) in enumerate(cross_val.split(features, target)):
            x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]

            model = ltb.LGBMRegressor(
                verbosity=-1, 
                random_state=self.random_state,  
                early_stopping_rounds=50,  # ✅ Reduzido para debug
                **hyperparams
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_test, y_test)],
                eval_metric="mse",
                callbacks=[LightGBMPruningCallback(trial, "l2")],
            )
            
            # ✅ DEBUG: Verificar treinamento
            preds = model.predict(x_test)
            cv_scores[idx] = mean_squared_error(y_test, preds)
            # print(f"Fold {idx} - MSE: {cv_scores[idx]:.4f}, Score: {model.score(x_test, y_test):.3f}")
        
        avg_score = np.mean(cv_scores)
        # print(f"Average CV score: {avg_score:.4f}")
        # print("======================")
        return avg_score

    def _hyperparam_tuning(self, features: pd.DataFrame, target: pd.Series) -> dict:
        """Use optuna for automating the hyperparameter tuning step"""
        # print("🚀 Starting hyperparameter tuning...")
        study = optuna.create_study(direction="minimize", study_name="Meta Model")
        func = lambda trial: self._objective(trial, features, target)
        study.optimize(func, n_trials=min(3, self.n_trials))  # ✅ Reduzido para debug
        
        # print(f"✅ Best hyperparams: {study.best_params}")
        # print(f"✅ Best value: {study.best_value:.4f}")
        return study.best_params
    
    def _get_n_most_important_features(self, model, n_features: int) -> list:
        # ✅ DEBUG: Verificar feature importances
        importances = np.array(model.feature_importances_, dtype=float)
        # print(f"📊 Feature importances - Non-zero: {np.sum(importances > 0)}/{len(importances)}")
        # print(f"📊 Max importance: {importances.max():.6f}")
        
        imp_df = pd.DataFrame({"name": model.feature_name_, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        return list(imp_df.head(n_features)["name"])

    def _select_features(self, features: pd.DataFrame, target: pd.Series=None) -> pd.DataFrame:
        if self.feature_list:
            return features[self.feature_list]

        # ✅ DEBUG: Verificar se feature selection é necessário
        # print(f"🔍 Feature selection: select_k_features={self.select_k_features}")
        
        if not self.select_k_features or self.select_k_features==1:
            self.feature_list = list(features.columns)
            # print(f"✅ Using all {len(self.feature_list)} features")
            return features

        if self.select_k_features < 1:
            n_features = int(np.ceil(features.shape[1] * self.select_k_features))
        else:
            n_features = self.select_k_features

        # ✅ DEBUG: Verificar dados antes do tuning
        # print(f"📈 Starting feature selection with {n_features} features")
        # print(f"📊 Features shape: {features.shape}, Target stats: std={target.std():.3f}")
        
        best_hyperparams = self._hyperparam_tuning(features, target)
        model = ltb.LGBMRegressor(**best_hyperparams).fit(features, target)
        
        # ✅ DEBUG: Verificar qualidade do modelo de feature selection
        train_score = model.score(features, target)
        # print(f"📊 Feature selection model score: {train_score:.3f}")
        
        self.feature_list = self._get_n_most_important_features(model, n_features)
        # print(f"✅ Selected {len(self.feature_list)} features")
        return features[self.feature_list]

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def fit(self, features: pd.DataFrame, target: pd.Series):
        """Fit meta model and do hyperparameter tuning with optuna."""
        # print("\n" + "="*50)
        # print("🎯 STARTING MetaModel.fit()")
        # print("="*50)
        
        # ✅ DEBUG: Verificar dados de entrada
        # print(f"📦 Input features shape: {features.shape}")
        # print(f"🎯 Target stats: min={target.min():.3f}, max={target.max():.3f}, std={target.std():.3f}")
        # print(f"🎯 Target unique values: {len(target.unique())}")
        
        # Feature selection
        features = self._select_features(features, target)
        # print(f"📋 Final features shape: {features.shape}")
        
        # Hyperparameter tuning
        if not self.best_hyperparams:
            print("🔄 Starting final hyperparameter tuning...")
            best_hyperparams = self._hyperparam_tuning(features, target)
            self.best_hyperparams = {
                "random_state": self.random_state, 
                "verbose": -1, 
                **best_hyperparams
            }
            # print(f"✅ Final best hyperparams: {self.best_hyperparams}")
        
        # Train final model
        # print("🏋️ Training final model...")
        self.model = ltb.LGBMRegressor(**self.best_hyperparams).fit(features, target)
        
        # # ✅ DEBUG CRÍTICO: Verificar modelo final
        # print("\n🔍 FINAL MODEL ANALYSIS:")
        # print(f"📊 Model type: {type(self.model)}")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            non_zero = np.sum(importances > 0)
            # print(f"📈 Feature importances: {non_zero}/{len(importances)} non-zero")
            # print(f"📈 Max importance: {importances.max():.6f}")
            if non_zero > 0:
                top_features = np.argsort(importances)[-5:][::-1]
                # print(f"🏆 Top 5 features: {top_features}")
        else:
            print("❌ No feature_importances_ available")
        
        # Verificar predições
        train_pred = self.model.predict(features)
        train_mse = mean_squared_error(target, train_pred)
        train_r2 = self.model.score(features, target)
        # print(f"📊 Train MSE: {train_mse:.4f}, R²: {train_r2:.3f}")
        # print(f"📊 Predictions range: [{train_pred.min():.3f}, {train_pred.max():.3f}]")
        
        # print("✅ Finished meta model training")
        # print("="*50 + "\n")
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        features = self._select_features(features)
        return self.model.predict(features)