from mfes_extractor import MfeExtractor
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator


MAX_CLUSTERS = 10

KMEANS_PARAMS = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

class KmeansMfesExtractor(MfeExtractor):
    def fit(self):
        return self
    

    def _get_size_dist_metrics(self,labels:np.ndarray):
        unique, counts = np.unique(labels, return_counts=True)
        counts = counts.astype(np.float64)
        num_items = len(labels)
        counts/=num_items
        max_size = np.max(counts)
        min_size = np.min(counts)
        mean_size = np.mean(counts)
        
        return (max_size, min_size,mean_size)
    
    def _get_compactness(self,df:pd.DataFrame,kmeans:KMeans)->int:
        n_clusters = kmeans.n_clusters
        labels = kmeans.labels_
        compactness = 0

        for i in range(n_clusters):
            centroid_coord = kmeans.cluster_centers_[i]
            distances = np.square(df[labels == i] - centroid_coord)
            distance_sum = np.sum(np.sqrt(distances.sum(axis=1)))
            compactness += distance_sum
        return compactness
    
    def _get_connectivity(self,df:pd.DataFrame,kmeans:KMeans)->int:
        labels = kmeans.labels_
        dists_matrix = euclidean_distances(df)
        np.fill_diagonal(dists_matrix, np.inf)
        nearest_indices = np.argmin(dists_matrix, axis=1)
        ans=0
        for i in range(df.shape[0]):
            ans+=(labels[i]!=labels[nearest_indices[i]])
        return ans/df.shape[0]
                
    def _train_kmeans(self,df:pd.DataFrame)-> (KMeans|int) :
        inertias = []
        models = []
        max_clusters = min(df.shape[0],MAX_CLUSTERS)
        for num_clusters in range(1,max_clusters+1):
            kmeans = KMeans(num_clusters,**KMEANS_PARAMS).fit(df)
            kmeans.fit(df)
            inertias.append(kmeans.inertia_)
            models.append(kmeans)
        knee = KneeLocator(range(1, max_clusters+1), inertias, curve="convex", direction="decreasing").knee
        return (models[knee-1],knee)


    def evaluate(self,df:pd.DataFrame)->dict:
        kmeans, knee = self._train_kmeans(df)

        max_size_dist, min_size_dist, mean_size_dist = self._get_size_dist_metrics(kmeans.labels_)
        return {
            'kmenas_compactness': self._get_compactness(df,kmeans),
            'kmeans_connectivity': self._get_connectivity(df,kmeans),
            'kmeans_n_iter': kmeans.n_iter_,
            'kmeans_n_clusters': kmeans.n_clusters,
            'kmeans_inertia': kmeans.inertia_,
            'kmeans_knee': knee,    
            'kmeans_min_size_dist': min_size_dist,    
            'kmeans_max_size_dist': max_size_dist,    
            'kmeans_mean_size_dist':mean_size_dist,
        }
    
    


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0], [16,3]])
    
    print(KmeansMfesExtractor().evaluate(X))
