import pandas as pd
import numpy as np

class MetaDataManager:
    def __init__(self):
        self.metabase = pd.DataFrame()
        pass

    def get_train_metabase(self)->pd.DataFrame:
        return self.metabase.drop([col for col in self.metabase.columns if col.startswith("meta_predict")])
    
    def set_init_df(self,df:pd.DataFrame)->None:
        self.metabase=df

    def set_pred(self, prediction, prediction_col:str)->None:
        self.metabase[prediction_col]=prediction