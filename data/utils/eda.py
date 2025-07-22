import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDA():
    @staticmethod
    def make(df, sample_size=5):
        """
        Realiza análise exploratória em um DataFrame pandas.
        
        Parâmetros:
            df (pd.DataFrame): DataFrame a ser analisado
            sample_size (int): Quantidade de linhas para mostrar nas amostras
        """
        print("="*80)
        print("ANÁLISE EXPLORATÓRIA DE DADOS")
        print("="*80)
        
        ## 1. Informações básicas do dataset
        print("\n1. INFORMAÇÕES BÁSICAS")
        print(f"Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print("\nTipos de dados:")
        print(df.dtypes.to_string())
        
        ## 2. Amostra dos dados
        print("\n2. AMOSTRA DOS DADOS")
        print("\nPrimeiras linhas:")
        print(df.head(sample_size).to_string())
        print("\nÚltimas linhas:")
        print(df.tail(sample_size).to_string())
        print("\nAmostra aleatória:")
        print(df.sample(sample_size).to_string())
        
        ## 3. Estatísticas descritivas
        print("\n3. ESTATÍSTICAS DESCRITIVAS")
        print("\nEstatísticas para colunas numéricas:")
        print(df.describe(include=[np.number]).to_string())
        print("\nEstatísticas para colunas categóricas:")
        try:
            print(df.describe(include=['object', 'category']).to_string())
        except:
            pass
        
        ## 4. Dados faltantes
        print("\n4. DADOS FALTANTES (NaN)")
        missing_data = df.isna().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Valores Faltantes': missing_data,
            '% Faltante': missing_percent.round(2)
        })
        print(missing_df[missing_df['Valores Faltantes'] > 0].to_string())
        
        ## 5. Análise de cardinalidade
        print("\n5. CARDINALIDADE DAS COLUNAS CATEGÓRICAS")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            print(f"\nColuna: {col}")
            print(f"Valores únicos: {df[col].nunique()}")
            print("Top 5 valores mais frequentes:")
            print(df[col].value_counts().head().to_string())
        
        ## 6. Visualizações básicas
        print("\n6. VISUALIZAÇÕES BÁSICAS")
        
        # Histogramas para colunas numéricas
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("\nHistogramas para colunas numéricas:")
            df[numerical_cols].hist(bins=20, figsize=(15, 10))
            plt.tight_layout()
            plt.show()
        
        # Gráfico de barras para colunas categóricas com baixa cardinalidade
        for col in categorical_cols:
            if df[col].nunique() <= 20:
                plt.figure(figsize=(10, 4))
                sns.countplot(data=df, x=col)
                plt.title(f'Distribuição de {col}')
                plt.xticks(rotation=45)
                plt.show()
        print("\nAnálise concluída!")
