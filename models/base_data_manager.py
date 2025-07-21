import pandas as pd
import numpy as np

class BaseDataManager():
    def __init__(self):
        self.new_target_prt:int=None
        self.df= pd.DataFrame()

    def set_init_df(self,df:pd.DataFrame)->None:
        if(not self.df.empty):
            raise Exception("Initial database for Base Data Manager already created")
        self.df = df.copy()
        self.new_target_prt=df.shape[0]

    def get_raw(self)->pd.DataFrame:
        return self.df.copy()