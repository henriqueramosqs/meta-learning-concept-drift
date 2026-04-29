import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

BASE_MODELS = [
    "RandomForestClassifier",
    "SVC",
    "LogisticRegression",
    "DecisionTreeClassifier"
]

PERFORMANCE_METRICS = [
    "f1-score", 
    "precision",
    "recall",
    "kappa"
]

METHODS =[
    "without_drift",
    "with_drift",
    "perfect",
    # "fernanda_st",
    "baseline",
    
]

COLORS = [
    "#eb5600ff", # orange
    "#1a9988ff", # green
    "#595959ff", # grey
    "#6aa4c8ff", # blue
    "#f1c232ff", # yellow
    ]

COLORS = {
    # "#d9d9d9", # grey
    "Perfect":"#f1c232ff",
    "$F_o$":  "#ffd7b3",
    "$F_d$":"#FF6347" # red
}
rename_map={
    "with_drift": "$F_d$",
    "without_drift": "$F_o$",
    "perfect": "Perfect"
}

#  Colocar o perfect [X]
#  Plotar gráficos [X]
#  Incluir métrticas da Fernanda [ ]
#  Todos os df's tem que ter o mesmo shape [ ]
#  Usar PyTest para gantir que:
#       A) Não existe outro base model com performance predita maior que o da pick [ ]
#       B) A performance do método é igual a performance da pick [ ]
#  Ver os casos onde as picks são diferentes, mas as performances não 

class BaseLevelEvaluator:

    def _get_model_name(self,str:str)->str:
        for model in BASE_MODELS:
            if(model in str):
                return model 
        raise("model not found")
    


    def drop_perfect(self):
        for metric in PERFORMANCE_METRICS:
            self.methods_perfomances[metric].drop(columns=["perfect_performance"],inplace=True)

    def _get_methods_picks(self):
        for metric, df in self.aux_df_by_performance_metric.items():
            for method in METHODS:
                if(method=="perfect"):
                    relevant_cols = [f"{model}_real_performance" for model in BASE_MODELS]
                else:
                    relevant_cols =[f"{model}_{metric}_prediction_{method}" for model in BASE_MODELS]
                df[f"{method}_pick"] = [self._get_model_name(x) for x in df[relevant_cols].idxmax(axis=1)]
                self.methods_perfomances[metric][f"{method}_performance"] =  df.apply(
                    lambda row: row[f"{row[f'{method}_pick']}_real_performance"], 
                    axis=1
                )

        picks_cols= [f"{method}_pick" for method in METHODS]


    
    def _get_fernanda_st(self):
        for base_model in BASE_MODELS:
            cur_loc = os.path.dirname(__file__)
           
            relative_path = f'../results/fernanda_st/results_dataframes/base_model: {base_model} - dataset: {self.dataset} - select_k_features: {self.feature_fraction}.csv'
            path = os.path.join(cur_loc,relative_path)
            cur_df = pd.read_csv(path)

            if self.expected_shape == (-1,-1):
                self.expected_shape = cur_df.shape
            else:
                if cur_df.shape != self.expected_shape:
                    raise ValueError(f"DataFrame shape mismatch for model {base_model}: expected {self.expected_shape}, got {cur_df.shape}")
            
            for metric in PERFORMANCE_METRICS:
               relevant_cols =[f"{metric}_pred_0_with_drift"]
               rename_map = {
                    f"{metric}_pred_0_with_drift":  f"{base_model}_{metric}_prediction_fernanda_st" , 
                    } 
               filtered_df = cur_df[relevant_cols].rename(columns=rename_map).fillna(0)
               self.aux_df_by_performance_metric[metric] = pd.concat(
                    [self.aux_df_by_performance_metric[metric], filtered_df], axis=1
                )

    def __init__(self, dataset, feature_fraction,dir):
        self.dataset = dataset
        self.feature_fraction = feature_fraction    
        self.expected_shape = (-1,-1)
        self.aux_df_by_performance_metric = {metric: pd.DataFrame() for metric in PERFORMANCE_METRICS}
        self.methods_perfomances = {metric: pd.DataFrame() for metric in PERFORMANCE_METRICS}
        self.dir = dir
        self.dataset=self.dataset[0].upper() + self.dataset[1:]
        for base_model in BASE_MODELS:
            cur_loc = os.path.dirname(__file__)
           
            relative_path = f'../results/{dir}/results_dataframes/base_model: {base_model} - dataset: {dataset} - select_k_features: {feature_fraction}.csv'
            path = os.path.join(cur_loc,relative_path)
            cur_df = pd.read_csv(path)

            if self.expected_shape == (-1,-1):
                self.expected_shape = cur_df.shape
            else:
                if cur_df.shape != self.expected_shape:
                    raise ValueError(f"DataFrame shape mismatch for model {base_model}: expected {self.expected_shape}, got {cur_df.shape}")
            
            for metric in PERFORMANCE_METRICS:
               relevant_cols =[f"{metric}",f"{metric}_pred_0_with_drift",f"{metric}_pred_0_without_drift",
                               f"last_{metric}" ]
               rename_map = {
                    f"{metric}":     f"{base_model}_real_performance"       ,
                    f"{metric}_pred_0_with_drift":  f"{base_model}_{metric}_prediction_with_drift" , 
                    f"{metric}_pred_0_without_drift": f"{base_model}_{metric}_prediction_without_drift"  ,
                    f"last_{metric}":       f"{base_model}_{metric}_prediction_baseline"  ,
                    } 
               filtered_df = cur_df[relevant_cols].rename(columns=rename_map).fillna(0)
               self.aux_df_by_performance_metric[metric] = pd.concat(
                    [self.aux_df_by_performance_metric[metric], filtered_df], axis=1
                )
            #    print(f"filtered_df.columns: {filtered_df.columns}")
        # print("aux_cols:",self.aux_df_by_performance_metric["kappa"].columns)
        # self._get_fernanda_st()
        self._get_methods_picks()
       

    def get_gain(self, metric:str):
        gain_df = pd.DataFrame()
        for method in METHODS:
            gain_df[f"{method}_gain"]= self.methods_perfomances[metric][f"{method}_performance"] - self.methods_perfomances[metric][f"baseline_performance"]
        gain_df = gain_df.cumsum()
        return gain_df
    
    def plot_charts(self):
        fig, axis = plt.subplots(nrows=2,ncols=2,figsize=(30, 22))
        axis_flat = axis.flatten()
        lines_labels=[]
        for metric, cur_ax in zip(PERFORMANCE_METRICS,axis_flat):
            gain_df = self.get_gain(metric)
            print("ds",self.dataset)
            print("shape", gain_df.shape)
            print(gain_df.iloc[-1,:])
            for method in METHODS:
                if(method=="baseline"):
                    continue
                cur_color = COLORS.get(rename_map[method], "black")
                line=cur_ax.plot(
                    gain_df[f"{method}_gain"],
                    label=rename_map[method],
                    color=cur_color,
                    linewidth=3,
                    zorder=2 )
                lines_labels.append(line)

                
                y_values = gain_df[f"{method}_gain"]
                x_values = range(len(y_values))

                cur_ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
                
            # cur_ax.legend(loc='upper left', fontsize='small')
            cur_ax.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)
            cur_ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=1) 
            cur_ax.spines['top'].set_visible(False)
            cur_ax.spines['right'].set_visible(False)
            cur_ax.set_facecolor('#FAFAFA')
            cur_ax.set_title(metric,fontsize=32) 
        handles, labels = axis_flat[0].get_legend_handles_labels()

        fig.legend(
            handles, 
            labels,
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(METHODS)-1,          
            fontsize=32,
            frameon=True,
            shadow=True
        )
        fig.suptitle(f"Cumulative gain for {self.dataset} dataset",fontsize=48)
        plt.tight_layout(rect=[0, 0.10, 1, 0.95])
        plt.show()
        fig.savefig(f'base_level_acc_gain_{self.dataset.lower()}.pdf',bbox_inches='tight')

if __name__ == "__main__":
    evaluator = BaseLevelEvaluator(dataset="airlines", feature_fraction=100) 
    evaluator.plot_comparison()

