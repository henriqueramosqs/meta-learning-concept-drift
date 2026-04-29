import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data.utils.eda import EDA 
import numpy as np


R_STATE = 1245
class MetaDataManager:

    """
    Manages the data at meta level 
    """
    
    def __init__(self,pca_n_components:int,target_cols,step):
        self.pca_n_components = pca_n_components # Number of principal components for dimensionality reduction
        self.metabase = pd.DataFrame()           # The main DataFrame storing all meta-level data
        self.new_target_ptr=0                    # Pointer to the next row where performance metrics (targets) should be written
        self.cur_batch_size=0                    # Counter for new instances added since the last raw batch was generated
        self.target_cols=[]                      # Stores the names of the target performance columns
        self.learning_window_size=1
        self.step=step

        
    def has_new_batch(self)-> bool:
        """
        Checks if enough new instances have been added to form a raw batch (based on 'step').
        """
        return self.cur_batch_size >= self.step
    
    def get_train_metabase(self)->pd.DataFrame:
        """
        Retrieves the metabase ready for training, excluding columns used for
        storing model predictions
        """
        ans =self.metabase.drop([col for col in self.metabase.columns if col.startswith("meta_predict")])
        return  ans
    
    def set_init_df(self,df:pd.DataFrame)->None:
        """
        Initializes the metabase with a starting DataFrame (offline phase).
        """
        self.metabase=df.copy()
        self.new_target_ptr =df.shape[0]

    def _reduce_dim(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Principal Component Analysis (PCA) for dimensionality reduction if enabled.
        Fits the PCA object on the first call.
        """
        if not self.pca_n_components:
            return df
        
        df_filled = df.fillna(df.mean()) 
        # df_filled = df.fillna(-9999) 
    
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler().fit(df_filled)
        
        df_scaled = self.scaler.transform(df_filled)
        # df_scaled = df_filled

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
        """
        Appends a new row to the metabase.
        """
        self.metabase= pd.concat([self.metabase,new_instance],axis=0, ignore_index=True)


    def update_target(self,target:dict)->None:
        """
        Updates the performance metrics (target values) for the instance pointed to by
        `self.new_target_ptr`
        """
        for key, value in target.items():
            self.metabase.at[self.new_target_ptr, key] = value

        self.new_target_ptr+=1
        self.cur_batch_size+=1

    def get_train_batch(self)->pd.DataFrame:
        """
        Retrieves the training batch using a sliding window of size `self.learning_window_size`.
        The window is defined by the interval [lower_bound, upper_bound).
        """

        lower_bound = self.new_target_ptr - self.learning_window_size
        upper_bound = self.new_target_ptr
        if(lower_bound<0):
            raise Exception("Not enough data to retireve a metabase batch")
        train_df = self.metabase.iloc[lower_bound:upper_bound].filter(regex='^(?!meta_predict_)')
        self.cur_batch_size=0
        return self._reduce_dim(pd.DataFrame(train_df))

    def get_raw(self)->pd.DataFrame:
        """
        Returns a copy of the entire raw metabase DataFrame.
        """
        return self.metabase.copy()
    
    def get_targeted_raw(self)->pd.DataFrame:
        """
        Returns the portion of the metabase that has already been assigned target values.
        """
        return self.get_raw()[:self.new_target_ptr]

    def set_pred(self, prediction, prediction_col:str)->None:
        """
        Sets the prediction values in a specified column in the metabase.
        """
        self.metabase[prediction_col]=prediction
        
    def get_last_tageted_row(self)->pd.Series:
        """
        Retrieves the row that was most recently updated with target performance metrics.
        """
        return self.metabase.iloc[self.new_target_ptr - 1]
    