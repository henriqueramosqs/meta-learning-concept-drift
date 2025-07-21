from .mfes_extractor import MfeExtractor
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

DBSCAN_PARAMS = {
    'eps': 0.3,
    'min_samples': 10,
}

class ClustringMetric():

    def _get_size_dist_metrics(self,labels:np.ndarray):
        unique, counts = np.unique(labels[labels!=-1], return_counts=True)
        counts = counts.astype(np.float64)
        num_items = len(labels[labels!=-1])
        counts/=num_items
        max_size = np.max(counts)
        min_size = np.min(counts)
        mean_size = np.mean(counts)
        
        return (max_size, min_size,mean_size)
    
    def _get_compactness(self,df:pd.DataFrame, labels:np, n_clusters:int, cluster_centers:np.ndarray)->int:
        compactness = 0

        for i in range(n_clusters):
            centroid_coord = cluster_centers[i]
            distances = np.square(df[labels == i] - centroid_coord)
            distance_sum = np.sum(np.sqrt(distances.sum(axis=1)))
            compactness += distance_sum
        return compactness
    
    def _get_connectivity(self,df:pd.DataFrame,labels:np.ndarray)->int:
        dists_matrix = euclidean_distances(df)
        np.fill_diagonal(dists_matrix, np.inf)
        nearest_indices = np.argmin(dists_matrix, axis=1)
        ans=0
        for i in range(df.shape[0]):
            ans+=(labels[i]!=labels[nearest_indices[i]])
        return ans/df.shape[0]