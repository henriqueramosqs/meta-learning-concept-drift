from .mfes_extractor import MfeExtractor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from .clustering_metric import ClustringMetric
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

DBSCAN_PARAMS = {
    'eps': 0.3,
    'min_samples': 5,
}

class DBSCANMfesExtractor(MfeExtractor,ClustringMetric):
    def fit(self):
        self.transform = None
        return self
    
    def _train(self,df:pd.DataFrame)-> (DBSCAN|int) :
        dbscan = DBSCAN(**DBSCAN_PARAMS).fit(df)
        return dbscan
    
    def _get_centroids(self,df:pd.DataFrame,dbscan:DBSCAN,labels:np.ndarray,n_clusters:int)->np.ndarray:
        centroids = []
        for i in range(0,n_clusters):
            cluster_points = df[labels == i]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        return np.array(centroids)


    def _handle_all_noise_case(self, df_shape):
        return {
            'dbscan_n_clusters': 0, 
            'dbscan_noise_proportion': 1.0,
            'dbscan_compactness': 0.0,
            'dbscan_connectivity': 0.0,
            'dbscan_min_size_dist': 0.0,    
            'dbscan_max_size_dist': 0.0,    
            'dbscan_mean_size_dist': 0.0,
        }


    def evaluate(self,df:pd.DataFrame)->dict:
        df = df.select_dtypes(include=np.number)
        df = StandardScaler().fit_transform(df)  
        dbscan = self._train(df)
        labels = dbscan.labels_
        n_clusters = len(set(labels) - {-1})
        noise_prop = list(labels).count(-1) / df.shape[0]
        if len(np.unique(labels)) == 1 and np.unique(labels)[0] == -1:
                return self._handle_all_noise_case(df.shape)
        
        cluster_centers = self._get_centroids(df,dbscan,labels,n_clusters)
        max_size_dist, min_size_dist, mean_size_dist = self._get_size_dist_metrics(labels)
        
        return {
            'dbscan_n_clusters': n_clusters, 
            'dbscan_noise_proportion': noise_prop,
            'dbscan_compactness': self._get_compactness(df,labels,n_clusters,cluster_centers),
            'dbscan_connectivity': self._get_connectivity(df,labels),
            'dbscan_min_size_dist': min_size_dist,    
            'dbscan_max_size_dist': max_size_dist,    
            'dbscan_mean_size_dist':mean_size_dist,
        }
    


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0], [16,3]])
    
    print(DBSCANMfesExtractor().evaluate(X))
