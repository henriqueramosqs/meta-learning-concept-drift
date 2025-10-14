import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA():

    """
    A utility class for performing standard Exploratory Data Analysis (EDA) 
    on a pandas DataFrame.
    """

    @staticmethod
    def lacking(df:pd.DataFrame,show_if_filled=False)->bool:

        """
        Calculates and prints the number and percentage of missing values (NaN) 
        per column. Returns True if there are missing values, False otherwise.
        
        Args:
            df (pd.DataFrame): The DataFrame to check.
            show_if_filled (bool): If True, shows the report even if no data is missing.
        
        Returns:
            bool: True if missing data was found, False otherwise.
        """
         
        missing_data = df.isna().sum()
        if missing_data.sum() == 0 and not show_if_filled:
            return False
        
        print("\n4. MISSING DATA (NaN)")
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Data': missing_data,
            '% Missing': missing_percent.round(2)
        })
        print(missing_df[missing_df['Missing Data'] > 0].to_string())
        return True
        
    @staticmethod
    def make(df, sample_size=5):
        """
        Orchestrates and runs the complete Exploratory Data Analysis process.
        
        Args:
            df (pd.DataFrame): The DataFrame to be analyzed.
            sample_size (int): Number of rows to display in sample outputs (head, tail, sample).
        """

        print("="*80)
        print("Exploratory Data Analysis")
        print("="*80)
        
        ## 1. Basic Dataset Information
        print("\n1. Basic Information")
        print(f"Dataset dimensions: {df.shape} ")
        print("\nData types:")
        print(df.dtypes.to_string())
        
        ## 2. Data sampling
        print("\n2. Data sampling")
        print("\nFirst lines:")
        print(df.head(sample_size).to_string())
        print("\nLast lines:")
        print(df.tail(sample_size).to_string())
        print("\nRandom sample:")
        print(df.sample(sample_size).to_string())
        
        ## 3. Descriptive statistics
        print("\n3. Descriptive statistices")
        print("\nNumerical columns statistics:")
        print(df.describe(include=[np.number]).to_string())
        print("\nCategorical columns statistics:")
        try:
            print(df.describe(include=['object', 'category']).to_string())
        except:
            pass
        
        ## 4. Missing data
        EDA.lacking(df)
       
        ## 5. Cardinality analysis
        print("\n5. Categorical columns cardinality:")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            print(f"\Column: {col}")
            print(f"unique values: {df[col].nunique()}")
            print("Most frequent values:")
            print(df[col].value_counts().head().to_string())
        
        ## 6.Basic visualizations
        print("\n6. VBasic visualizations")
        
        # Histograms for numerical columns 
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("\nHistograms for numerical columns :")
            df[numerical_cols].hist(bins=20, figsize=(15, 10))
            plt.tight_layout()
            plt.show()
        
        # Bar chart fot columns with low cardinality
        for col in categorical_cols:
            if df[col].nunique() <= 20:
                plt.figure(figsize=(10, 4))
                sns.countplot(data=df, x=col)
                plt.title(f'{col} distribution')
                plt.xticks(rotation=45)
                plt.show()
        print("\nAnalysis finished!")
