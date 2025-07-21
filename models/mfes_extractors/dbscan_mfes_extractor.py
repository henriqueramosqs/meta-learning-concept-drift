from .mfes_extractor import MfeExtractor
from sklearn.metrics.pairwise import euclidean_distances
from .clustering_metric import ClustringMetric
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

DBSCAN_PARAMS = {
    'eps': 0.3,
    'min_samples': 10,
}

class DBSCANMfesExtractor(MfeExtractor,ClustringMetric):
    def fit(self):
        return self
    
    def _train(self,df:pd.DataFrame)-> (DBSCAN|int) :
        df_size = df.shape[0]
        dbscan = DBSCAN(**DBSCAN_PARAMS).fit(df)
        return dbscan
    
    def _get_centroids(self,df:pd.DataFrame,dbscan:DBSCAN,labels:np.ndarray,n_clusters:int)->np.ndarray:
        centroids = []
        for i in range(0,n_clusters):
            cluster_points = df[labels == i]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        return np.array(centroids)

    def evaluate(self,df:pd.DataFrame)->dict:
        dbscan = self._train(df)
        labels = dbscan.labels_
        n_clusters = len(set(labels) - {1})
        noise_prop = list(labels).count(-1) / df.shape[0]
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
