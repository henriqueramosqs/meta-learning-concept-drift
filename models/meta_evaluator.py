import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from models import Evaluator
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data.utils.eda import EDA

# Macros
COLORS = [
    "#eb5600ff", # orange
    "#1a9988ff", # green
    "#595959ff", # grey
    "#6aa4c8ff", # blue
    "#f1c232ff", # yellow
    ]
BASE_MODELS = [
    "RandomForestClassifier",
    "SVC",
    "LogisticRegression",
    "DecisionTreeClassifier"
]


class MetaEvaluator():
    def __init__(self, dataset_name: str, window_size: int = 30,dir=None,feature_fraction=100):
        self.window_size = window_size
        self.dataset_name = dataset_name
        self.feature_fraction = feature_fraction
        self.dir = dir

    def _get_mean_mse(self, cols: list, data_frame: pd.DataFrame, metric: str=None):
        result_mse = pd.DataFrame(columns=cols).astype(float)
       
        if not metric:
            metric = cols[0].split("_")[0]
            # print(f"Caso em que não tem espercificado: metric:{metric} cols:{cols}")

        iter_range = range(0, data_frame.shape[0] - self.window_size, self.window_size)

        for result_idx, original_idx in enumerate(iter_range):
            batch = data_frame.iloc[original_idx:original_idx + self.window_size]
            for col in cols:
                result_mse.loc[result_idx, col] = mean_squared_error(batch[metric], batch[col])

        # print("result_mse")    
        # print(result_mse.shape)
        # print(result_mse.info())
        return result_mse.mean(axis=1)


    def _get_result_df(self, filename: str):
        notebook_dir = os.path.dirname(os.path.abspath('.'))  
        correct_path = os.path.join(notebook_dir,filename)

        if not os.path.exists(correct_path):
            print(f"ERRO: Arquivo não encontrado em {correct_path}")
    
        df = pd.read_csv(correct_path).dropna()

        metrics = list(set(df.columns).intersection(set(["auc","f1-score","recall", "precision", "kappa"])))


        # print(f"df_head: {filename}")
        # print(df.head())
        # print("="*50)

        auxx = df[[f"last_{metric}" for metric in metrics]]
        # print("Esse é o auxx")
        # print(auxx)
        results = pd.DataFrame()
        for metric in metrics:
            metric_cols = [col for col in df.columns if metric in col]
            with_drift_cols = [col for col in metric_cols if "with_drift" in col]
            without_drift_cols = [col for col in metric_cols if "without_drift" in col]

            # print("with_drift_cols: ", with_drift_cols)
            # print("without_drift_cols", without_drift_cols)
            # print("metric_cols",  metric_cols)

            results[f"{metric}_mse_with_drift"] = self._get_mean_mse(with_drift_cols, df)
            results[f"{metric}_mse_without_drift"] = self._get_mean_mse(without_drift_cols, df)
            results[f"{metric}_mse_baseline"] = self._get_mean_mse([f"last_{metric}"], df, metric)
        return results, metrics

    def _plot_subplot(self, results_df: pd.DataFrame, color: str=COLORS[0], metric="kappa"):
        mtl_with_drift_error = results_df[f"{metric}_mse_with_drift"]
        baseline_error  = results_df[f"{metric}_mse_baseline"]
        mtl_without_drift_error = results_df[f"{metric}_mse_without_drift"]
        mtl_with_drift_gain =  -mtl_with_drift_error + baseline_error

        y = mtl_with_drift_gain.cumsum()

        print(f"y[-1]: {y.iloc[-1]}")
        x = np.arange(len(y))
        plt.fill_between(x, y, alpha=0.1, color=color)
        plt.plot(x, y, label=metric, color=color)
        plt.legend(loc=2, fontsize='large')

    def fit(self):
        self.results = {}
        self.metrics = {}
        for base_model in BASE_MODELS:
            filename = f"results/{self.dir}/results_dataframes/base_model: {base_model} - dataset: {self.dataset_name} - select_k_features: {self.feature_fraction}.csv"
            self.results[base_model], self.metrics[base_model] = self._get_result_df(filename)
            # print(filename)
            # print(self.results[base_model].head(5))
            # print(self.metrics[base_model])
            # print("="*50)

        return self

    def plot_gain(self):
        plt.figure(figsize=(25, 15))
        # plt.ylim(bottom=0)
        plt.suptitle(f"Ganho com dataset: {self.dataset_name}", fontsize=25)
        show = True

        # print(f"self.results. {self.results.shape}")
        # print(f"self.results. {self.results.info()}")
        for base_model_idx, base_model in enumerate(BASE_MODELS):
            plt.subplot(2, 2, base_model_idx + 1)
            for metric_idx, metric in enumerate(self.metrics[base_model]):
                plt.title(f"Base_model: {base_model}", fontsize=20)

                print(f"Base_model: {base_model}, metric: {metric}")
                # print(self.results[base_model].head())
                #
                self._plot_subplot(self.results[base_model], metric=metric, color=COLORS[metric_idx])
                print("="*60)


    def _plot_comp_subplot(self, results_df: pd.DataFrame, color: str=COLORS[0], 
                        metric: str="kappa", plot_col: str="proposed_mtl"):
        
        if plot_col == "proposed_mtl":
            regressor_error = results_df[f"{metric}_mse_with_drift"]
            baseline_error = results_df[f"{metric}_mse_baseline"]
            gain = baseline_error - regressor_error
            
        elif plot_col == "original_mtl":
            regressor_error = results_df[f"{metric}_mse_without_drift"]
            baseline_error = results_df[f"{metric}_mse_baseline"]
            gain = baseline_error - regressor_error
            
        elif plot_col == "ideal_regressor":
            baseline_error = results_df[f"{metric}_mse_baseline"]
            gain = baseline_error  # Pois ideal teria erro 0
        
        y = gain.cumsum()
        x = np.arange(len(y))
        plt.fill_between(x, y, alpha=0.1, color=color)
        plt.plot(x, y, label=plot_col, color=color)
        plt.legend(loc=2, fontsize='large')

    def plot_original_vs_proposed_mtl_gain(self, metric="kappa", plot_ideal_regressor=True, subplot_index=1):

        for base_model in BASE_MODELS:

            # print(f"Base_model: {base_model}")
            # print(self.results[base_model].head())
            plt.subplot(4, 4, subplot_index)
            if plot_ideal_regressor:
                self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[2], plot_col="ideal_regressor")
            self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[0], plot_col="proposed_mtl")
            self._plot_comp_subplot(self.results[base_model], metric=metric, color=COLORS[1], plot_col="original_mtl")
            subplot_index += 1
            plt.title(f"{metric} - {base_model}")

    def verificar_igualdade_metricas(self, base_model=None):
        """
        Verifica se os valores de recall, precision e f1-score são iguais
        para os diferentes tipos de regressores.
        
        Parameters:
        -----------
        base_model : str, optional
            Nome do modelo base a ser verificado. Se None, verifica todos os modelos.
        
        Returns:
        --------
        dict
            Dicionário com os resultados da verificação
        """
        
        resultados = {}
        modelos_verificar = [base_model] if base_model else BASE_MODELS
        
        for modelo in modelos_verificar:
            df = self.results[modelo]
            resultados[modelo] = {
                'proposed_mtl': {
                    'recall_vs_precision': df['recall_mse_with_drift'].equals(df['precision_mse_with_drift']),
                    'recall_vs_f1': df['recall_mse_with_drift'].equals(df['f1-score_mse_with_drift']),
                    'precision_vs_f1': df['precision_mse_with_drift'].equals(df['f1-score_mse_with_drift']),
                    'todos_iguais': (df['recall_mse_with_drift'].equals(df['precision_mse_with_drift']) and
                                df['recall_mse_with_drift'].equals(df['f1-score_mse_with_drift']))
                },
                'original_mtl': {
                    'recall_vs_precision': df['recall_mse_without_drift'].equals(df['precision_mse_without_drift']),
                    'recall_vs_f1': df['recall_mse_without_drift'].equals(df['f1-score_mse_without_drift']),
                    'precision_vs_f1': df['precision_mse_without_drift'].equals(df['f1-score_mse_without_drift']),
                    'todos_iguais': (df['recall_mse_without_drift'].equals(df['precision_mse_without_drift']) and
                                df['recall_mse_without_drift'].equals(df['f1-score_mse_without_drift']))
                },
                'baseline': {
                    'recall_vs_precision': df['recall_mse_baseline'].equals(df['precision_mse_baseline']),
                    'recall_vs_f1': df['recall_mse_baseline'].equals(df['f1-score_mse_baseline']),
                    'precision_vs_f1': df['precision_mse_baseline'].equals(df['f1-score_mse_baseline']),
                    'todos_iguais': (df['recall_mse_baseline'].equals(df['precision_mse_baseline']) and
                                df['recall_mse_baseline'].equals(df['f1-score_mse_baseline']))
                }
            }
        
        return resultados

    def exibir_resultados_verificacao(resultados):
        """
        Exibe os resultados da verificação de forma organizada.
        """
        for modelo, dados_modelo in resultados.items():
            print(f"\n=== MODELO: {modelo} ===")
            
            for tipo_regressor, comparacoes in dados_modelo.items():
                print(f"\n{tipo_regressor.upper()}:")
                print(f"  Recall vs Precision: {'IGUAIS' if comparacoes['recall_vs_precision'] else 'DIFERENTES'}")
                print(f"  Recall vs F1-score: {'IGUAIS' if comparacoes['recall_vs_f1'] else 'DIFERENTES'}")
                print(f"  Precision vs F1-score: {'IGUAIS' if comparacoes['precision_vs_f1'] else 'DIFERENTES'}")
                print(f"  Todos iguais: {'SIM' if comparacoes['todos_iguais'] else 'NÃO'}")

    def verificar_baseline_metrics(self):
        """
        Verifica se os valores de baseline são iguais entre as diferentes métricas.
        """
        resultados = {}
        
        for base_model in BASE_MODELS:
            df = self.results[base_model]
            modelo_resultados = {}
            
            # Pegar todas as colunas de baseline
            baseline_cols = [col for col in df.columns if 'mse_baseline' in col]
            metricas = [col.replace('_mse_baseline', '') for col in baseline_cols]
            
            # Verificar se todas as colunas de baseline são iguais
            todas_iguais = True
            for i in range(len(baseline_cols)):
                for j in range(i + 1, len(baseline_cols)):
                    if not df[baseline_cols[i]].equals(df[baseline_cols[j]]):
                        todas_iguais = False
                        break
                if not todas_iguais:
                    break
            
            if todas_iguais:
                modelo_resultados['todas_metricas_iguais'] = True
                modelo_resultados['valor_comum'] = df[baseline_cols[0]].iloc[0] if len(df) > 0 else None
            else:
                # Verificar pares específicos que são iguais
                pares_iguais = []
                for i in range(len(baseline_cols)):
                    for j in range(i + 1, len(baseline_cols)):
                        if df[baseline_cols[i]].equals(df[baseline_cols[j]]):
                            par = f"{metricas[i]} == {metricas[j]}"
                            pares_iguais.append(par)
                
                if pares_iguais:
                    modelo_resultados['pares_iguais'] = pares_iguais
            
            if modelo_resultados:
                resultados[base_model] = modelo_resultados
        
        return resultados

    def exibir_resultados_baseline(self, resultados):
        """
        Exibe os resultados da verificação dos baselines.
        """
        if not resultados:
            print("Nenhum caso onde os baselines são iguais entre métricas diferentes.")
            return
        
        for modelo, info in resultados.items():
            print(f"\n=== MODELO: {modelo} ===")
            
            if 'todas_metricas_iguais' in info:
                print("✓ TODAS as métricas de baseline são iguais!")
                print(f"  Valor comum: {info['valor_comum']}")
            elif 'pares_iguais' in info:
                print("Alguns pares de métricas têm baseline igual:")
                for par in info['pares_iguais']:
                    print(f"  ✓ {par}")