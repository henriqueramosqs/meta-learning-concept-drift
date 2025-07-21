import pandas as pd
import numpy as np

class BaseDataManager():
    def __init__(self,batch_size:int):
        self.batch_size:int=batch_size
        self.new_target_prt:int=None
        self.df= pd.DataFrame()
        self.cur_batch_size:int =0
        self.cur_targeted_batch_size:int=0

    def set_init_df(self,df:pd.DataFrame)->None:
        if(not self.df.empty):
            raise Exception("Initial database for Base Data Manager already created")
        self.df = df.copy()
        self.new_target_prt=df.shape[0]

    def get_raw(self)->pd.DataFrame:
        return self.df.copy()
    
    def update(self,new_instance: pd.DataFrame)-> None:
        self.cur_batch_size+=1
        self.df= pd.concat([self.df,new_instance],axis=0)
    
    def get_targeted_batch(self)->pd.DataFrame:
        res_df = self.df.dropna("class", axis=0)
        if(res_df.shape[0]<self.batch_size):
            raise Exception("There's no enough targeted data to compose a batch in the base data manager")
        self.cur_targeted_batch_size=0
        return res_df.tail(self.batch_size)  
    
    def get_last_batch(self) -> pd.DataFrame:
        if(self.df.shape[0]<self.batch_size):
            raise Exception("There's no enough data to compose a batch in the base data manager")
        self.cur_batch_size=0
        return self.df.drop("class", axis=1).tail(self.batch_size)  

    def update_target(self,target:float)->None:
        self.df.at[self.new_target_prt, "class"]=target
        self.new_target_prt+=1
        self.cur_targeted_batch_size+=1
    
    def has_new_batch(self):
        return self.cur_batch_size > self.batch_size
    
    def has_new_targeted_batch(self):
        return self.cur_batch_size > self.batch_size