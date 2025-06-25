import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
import os

class DataLoader:
    def load_data(src:str)->pd.DataFrame:
        cur_dir = os.path.dirname(__file__)
        file_dir =os.path.join(cur_dir, "datasets",src)
        file = open(file_dir, "r", encoding="utf-8")
        data, meta_data = arff.loadarff(file)
        if(not "class" in meta_data.names()):
            raise Exception(f"{src} dataset  does not feature a target column named \"class\"")
        df = pd.DataFrame(data)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"Loading dataset {src}")
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
            df[col]=df[col].astype('category')

        print(f"Performing Label Encoding on the following columns: {cat_cols}")
        for col in df.columns:
            if(df[col].dtypes=='category'):
                print(col,": " ,df[col].unique(),"\n")
        
        print(f"Returning dataframe with success\n")
        return df



if __name__ == "__main__":  
    DataLoader.load_data("real/rialto.arff")
    DataLoader.load_data("real/electricity.arff")
    DataLoader.load_data("real/airlines.arff")
    DataLoader.load_data("real/Powersupply.arff")
    