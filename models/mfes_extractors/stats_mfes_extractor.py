from .mfes_extractor import MfeExtractor
from scipy.stats import hmean, gmean, entropy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from kneed import KneeLocator

# Usado para numero de atributos com alta correlação
TAU = 0.5
PCA_THRESH = 0.95

class StatsMFesExtractor(MfeExtractor):
    def fit(self):
        return self
    
    def _get_iqr_metrics(self,df:pd.DataFrame):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        nr_out = ((df<(q1-1.5*iqr)) | (df>(q3+1.5*iqr))).sum()
        nr_out = {f'nr_outliers_{key}':value for key, value in nr_out.to_dict().items()}
        iqr_dict = {f'iqr_{key}':value for key, value in iqr.to_dict().items()}
        return {**nr_out, **iqr_dict}
    
    def _get_correlation(self,df:pd.DataFrame)->pd.DataFrame:
        corr = df.corr()
        mask = np.tril(corr,k=-1).astype(bool)
        d = corr.shape[0]
        nrCorAttr =(2/((d)*(d-1)))*corr.where(mask & (corr > TAU)).values.sum()
        ans = { 
            f'corr_{corr.columns[i]}_{corr.columns[j]}': corr.iloc[i, j]
            for i, j in zip(*np.where(mask))
            }
        ans['nrCorAttr']=nrCorAttr
        return ans
    
    def _get_sparsity(self,df:pd.DataFrame)->float:
        num_instances,num_features = df.shape
        unique_vals, phi_i = np.unique(df, return_counts=True)
        phi_x = len(unique_vals)
        sum= (phi_i.sum() / phi_x)
        return  (sum-1)/(num_instances-1)
        
    
    def evaluate(self,df:pd.DataFrame)->dict:
        df = df.select_dtypes(include=np.number)
        num_instances,num_features = df.shape
        max_dict =  {f'max_{key}': value for key, value in df.max().to_dict().items()}
        min_dict =  {f'min_{key}': value for key, value in df.min().to_dict().items()}
        mean_dict =  {f'mean_{key}': value for key, value in df.mean().to_dict().items()}
        median_dict =  {f'median_{key}': value for key, value in df.median().to_dict().items()}
        std_dict =  {f'std_{key}': value for key, value in df.std().to_dict().items()}
        var_dict =  {f'var_{key}': value for key, value in df.var().to_dict().items()}
        kurt_dict =  {f'kurtosis_{key}': value for key, value in df.kurt().to_dict().items()}
        skew_dict =  {f'skewness_{key}': value for key, value in df.skew().to_dict().items()}
        corr_dict = self._get_correlation(df)
        gmean_dict = {f'gmean_{column}': gmean(df[column]) for column in df.columns}
        hmean_dict = {f'hmean_{column}': hmean(df[column]) for column in df.columns}
        entropy_dict = {f'entropy_{column}': entropy(df[column].value_counts(normalize=True)) for column in df.columns}
        pca_dict = {'prop_pca': PCA(n_components=PCA_THRESH).fit_transform(df).shape[1]/num_features}
        iqr_dict = self._get_iqr_metrics(df)
        uniqueness_dict = {f'uniqueness_ratio_{key}': value for key,value in (df.nunique()/num_instances).to_dict().items()}
        sparsity_dict = {f'sparsity_{key}': value for key,value in (1 - df.notna().sum(axis=0) /num_instances).to_dict().items()}
        attr_sparsity_dict = {f'attr_sparsity_': self._get_sparsity(df)}
        
        return {
            **max_dict, 
            **min_dict, 
            **mean_dict,
            **median_dict,
            **std_dict,
            **var_dict,
            **kurt_dict,
            **skew_dict,
            **corr_dict,
            **gmean_dict,
            **hmean_dict,
            **entropy_dict,
            **pca_dict,
            **iqr_dict,
            **uniqueness_dict,
            **sparsity_dict,
            **attr_sparsity_dict
        }

if __name__ == "__main__":  
    df = pd.DataFrame({'a': [1, 2, 2, 3], 'b': [3, 4, 4, 4]},
                  index=['cat', 'dog', 'dog', 'mouse'])
    aux = StatsMFesExtractor().evaluate(df)
    print(aux)