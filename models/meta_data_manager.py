import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data.utils.eda import EDA 
import numpy as np


R_STATE = 1245
class MetaDataManager:
    def __init__(self,pca_n_components:int,target_cols):
        self.pca_n_components = pca_n_components
        self.metabase = pd.DataFrame()
        self.new_target_ptr=0
        self.cur_batch_size=0
        self.target_cols=[]
        pass

    def get_train_metabase(self)->pd.DataFrame:
        return self.metabase.drop([col for col in self.metabase.columns if col.startswith("meta_predict")])
    
    def set_init_df(self,df:pd.DataFrame)->None:
        self.metabase=df.copy()
        self.new_target_ptr = df.shape[0]

    def _reduce_dim(self,df: pd.DataFrame) -> pd.DataFrame:
        if not self.pca_n_components:
            return df
        
        df_filled = df.fillna(df.mean()) 
    
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler().fit(df_filled)
        
        df_scaled = self.scaler.transform(df_filled)

        if not hasattr(self, 'pca'):
            svd_solver = "auto" if self.pca_n_components > 1 else "full"
            self.pca = PCA(
                n_components=self.pca_n_components,
                svd_solver=svd_solver,
                random_state=R_STATE
            ).fit(df_scaled)

        n_comp = self.pca.n_components_
        variance = sum(self.pca.explained_variance_ratio_) * 100
        print(f"Dim reduction - keeping {n_comp} components explaining {variance:.2f}% of variance")
        return pd.DataFrame(self.pca.transform(df_scaled), self.pca)
          
    def update(self,new_instance:pd.DataFrame)->None:
        self.metabase= pd.concat([self.metabase,new_instance],axis=0)


    def update_target(self,target:dict)->None:
        for key, value in target.items():
            self.metabase.at[self.new_target_ptr, key] = value
        self.new_target_ptr+=1
        self.cur_batch_size+=1

    def get_train_batch(self)->pd.DataFrame:
        lower_bound = self.new_target_id - self.learning_window_size
        upper_bound = self.new_target_id
        if(lower_bound<0):
            raise Exception("Not enough data to retireve a metabase batch")
        train_df = self.metabase.iloc[lower_bound:upper_bound].filter(regex='^(?!meta_predict_)')
        self.cur_batch_size=0
        return self._reduce_dim(pd.DataFrame(train_df))

    def get_raw(self)->pd.DataFrame:
        return self.metabase.copy()
    
    def get_targeted_raw(self)->pd.DataFrame:
        return self.get_raw()[:self.new_target_ptr]

    def set_pred(self, prediction, prediction_col:str)->None:
        self.metabase[prediction_col]=prediction
    
    def get_last_tageted_row(self)->pd.Series:
        return self.metabase.iloc[self.new_target_ptr - 1]
    
    def exploratory_data_analysis(self):
        EDA.exploratory_data_analysis(self.df)