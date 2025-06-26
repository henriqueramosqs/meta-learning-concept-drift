from mfes_extractor import MfeExtractor
from clustering_metric import ClustringMetric
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

class KmeansMfesExtractor(MfeExtractor,ClustringMetric):
    def fit(self):
        return self
    
    def _train(self,df:pd.DataFrame)-> (KMeans|int) :
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
        df = df.select_dtypes(include=np.number)
        kmeans, knee = self._train(df)
        labels = kmeans.labels_
        n_clusters = kmeans.n_clusters
        cluster_centers = kmeans.cluster_centers_
        max_size_dist, min_size_dist, mean_size_dist = self._get_size_dist_metrics(labels)
   
        return {
            'kmeans_n_iter': kmeans.n_iter_,
            'kmeans_n_clusters': n_clusters,
            'kmeans_inertia': kmeans.inertia_,
            'kmeans_knee': knee,    
            'kmenas_compactness': self._get_compactness(df,labels,n_clusters,cluster_centers),
            'kmeans_connectivity': self._get_connectivity(df,labels),
            'kmeans_min_size_dist': min_size_dist,    
            'kmeans_max_size_dist': max_size_dist,    
            'kmeans_mean_size_dist':mean_size_dist,
        }
    
    


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0], [16,3]])
    
    print(KmeansMfesExtractor().evaluate(X))
