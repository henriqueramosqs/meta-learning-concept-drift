import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
import os

class DataLoader:

    """
    A utility class responsible for loading ARFF datasets, validating 
    the target column, and performing pre-processing Label encoding on  
    categorical features.
    """
    @staticmethod
    def load_data(src:str)->pd.DataFrame:
        cur_dir = os.path.dirname(__file__)
        file_dir =os.path.join(cur_dir, "datasets",src)
        file = open(file_dir, "r", encoding="utf-8")
        data, meta_data = arff.loadarff(file)
        if(not "class" in meta_data.names()):
            raise Exception(f"{src} dataset  does not feature a target column named \"class\"")
        df = pd.DataFrame(data)

        # Tirar depois. Isso é aplicável apenas para o Electricity
        # df['weekday'] = pd.to_numeric(df['day'])
        # df = df.drop(['date', 'period', 'day'], axis=1)

        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()


        print(f"Loading dataset {src}")
        print(df.info())
        print(f"Performing Label Encoding on the following columns: {cat_cols}")
    
        

        for col in df.columns:
            if(df[col].dtypes=='category' or df[col].dtypes=='object'):
                    print(col,": " ,df[col].unique(),"\n")
        
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        
        print(f"Returning dataframe with success\n")
        print(df.info())
        return df



if __name__ == "__main__":  
    df = DataLoader.load_data("real/Rialto.arff")
    EDA.lacking(df,True)
    DataLoader.load_data("real/electricity.arff")
    DataLoader.load_data("real/airlines.arff")
    DataLoader.load_data("real/Powersupply.arff")
    