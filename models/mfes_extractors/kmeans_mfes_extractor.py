from .mfes_extractor import MfeExtractor
from .clustering_metric import ClustringMetric
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Maximum number of clusters to test for the Elbow method
MAX_CLUSTERS = 10

# Default parameters for KMeans clustering
KMEANS_PARAMS = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

class KmeansMfesExtractor(MfeExtractor,ClustringMetric):
    """
    A meta-feature extractor that applies KMeans clustering
    and calculates various clustering-based meta-features.
    """

    def fit(self):
        """
        Method to adjust to inheritane from MfeExtractor
        """
        return self
    
    
    def _train(self,df:pd.DataFrame)-> (KMeans|int) :
        """
        Trains multiple KMeans models (k=1 to max_clusters) and selects the optimal
        number of clusters using the Elbow method (KneeLocator).

        Args:
            df (np.ndarray): The scaled data used for clustering.

        Returns:
            Tuple[KMeans, int]: The best fitted KMeans model and the determined optimal knee.
        """
        inertias = []
        models = []
        max_clusters = min(df.shape[0],MAX_CLUSTERS)
        for num_clusters in range(1,max_clusters+1):
            kmeans = KMeans(num_clusters,**KMEANS_PARAMS).fit(df)
            kmeans.fit(df)
            inertias.append(kmeans.inertia_)
            models.append(kmeans)
        knee = KneeLocator(range(1, max_clusters+1), inertias, curve="convex", direction="decreasing").knee
        if(knee is None):
            knee = max_clusters
        return (models[knee-1],knee)


    def evaluate(self,df:pd.DataFrame)->dict:
        """
        Performs standardization, applies the Kmeans the elbow method, and calculates clustering meta-features.

        Args:
            df (pd.DataFrame): The input data

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the KMeans-based meta-features.
        """
        
        df = df.select_dtypes(include=np.number)

        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler().fit(df)
        
        df = self.scaler.transform(df)

        kmeans, knee = self._train(df)
        labels = kmeans.labels_
        n_clusters = kmeans.n_clusters
        cluster_centers = kmeans.cluster_centers_
        max_size_dist, min_size_dist, mean_size_dist = self._get_size_dist_metrics(labels)
   
        return {
            'kmeans_n_iter': kmeans.n_iter_,
            'kmeans_n_clusters': n_clusters,
            'kmeans_inertia': kmeans.inertia_,
            # 'kmeans_knee': knee,    
            'kmenas_compactness': self._get_compactness(df,labels,n_clusters,cluster_centers),
            # 'kmeans_connectivity': self._get_connectivity(df,labels),
            # 'kmeans_min_size_dist': min_size_dist,    
            # 'kmeans_max_size_dist': max_size_dist,    
            # 'kmeans_mean_size_dist':mean_size_dist,
        }
    
    


if __name__ == "__main__":
    X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0], [16,3]])
    
    print(KmeansMfesExtractor().evaluate(X))
