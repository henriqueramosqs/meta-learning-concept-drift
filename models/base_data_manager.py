import pandas as pd
import numpy as np
from data.utils.eda import EDA 

class BaseDataManager():

    """
    Manages the base level data, maintaining a main DataFrame and controlling the size 
    of data batches for operations.
    """

    def __init__(self,batch_size:int, step:int):
        self.batch_size:int=batch_size
        self.step =step               # The minimum number of new instances/targeted instances required before a new batch is ready
        self.new_target_ptr:int=0     # Pointer to the next row to receive a 'class' (target)
        self.df= pd.DataFrame()       # The main DataFrame entries
        self.cur_batch_size:int =0    # Counter for new instances added since the last raw batch was generated
        self.cur_targeted_batch_size:int=0  # Counter for new instances that have received (target) since the last targeted batch was generated

    def set_init_df(self,df:pd.DataFrame)->None:
        """
        Initializes the main DataFrame with a starting dataset (offline phase)
        """
        if(not self.df.empty):
            raise Exception("Initial database for Base Data Manager already created")
        self.df = df.copy()
        self.new_target_ptr=df.shape[0]

    def get_raw(self)->pd.DataFrame:
        """
        Returns a copy of the entire raw DataFrame.
        """
        return self.df.copy()
    
    def update(self,new_instance: pd.DataFrame)-> None:
        """
        Adds a new row to the main DataFrame.
        """
        self.cur_batch_size+=1
        self.df= pd.concat([self.df,new_instance],axis=0,  ignore_index=True)
    
    def get_targeted_batch(self)->pd.DataFrame:
        """
        Retrieves the latest batch of target data
        """
        res_df = self.df.dropna(subset=['class'])
        if(res_df.shape[0]<self.batch_size):
            raise Exception("There's no enough targeted data to compose a batch in the base data manager")
        self.cur_targeted_batch_size=0
        return res_df.tail(self.batch_size)  
    
    def get_last_batch(self) -> pd.DataFrame:
        """
        Retrieves the latest batch of data (with or without target valuues)
        """
        if(self.df.shape[0]<self.batch_size):
            raise Exception("There's no enough data to compose a batch in the base data manager")
        self.cur_batch_size=0
        return self.df.drop("class", axis=1).tail(self.batch_size)  

    def update_target(self,target:dict)->None:
        """
        Assigns a target value to the instance at the 'new_target_ptr' 
        and advances the pointer.
        """
        self.df.at[self.new_target_ptr, "class"]=target
        self.new_target_ptr+=1
        self.cur_targeted_batch_size+=1

    
    def has_new_batch(self)-> bool:
        """
        Checks if enough new instances have been added to form a raw batch (based on 'step').
        """
        return self.cur_batch_size >= self.step
    
    def has_new_targeted_batch(self)->bool:
        """
        Checks if enough new labeled instances have been added to form a targeted batch (based on 'step').
        """
        return self.cur_targeted_batch_size >= self.step
    
    def get_last_tageted_row(self)->pd.Series:
        """
        Retrieves the row that was most recently updated with target performance metrics.
        """
        return self.df.iloc[self.new_target_ptr - 1]
    