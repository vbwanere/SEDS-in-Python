from functools import partial
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):

        self.n_clusters = n_clusters # k
        self.max_iter = max_iter
        self.centroids = None # k-means

    def fit_predict(self, X):

        # initialize any k number of data-points as centroids:
        self.centroids = X[:self.n_clusters] 

        for i in range(self.max_iter):
            # assign clusters
            cluster_indices = self.assign_clusters(X)
            old_centroids = self.centroids
            # update centroids
            self.centroids = self.move_centroids(X, cluster_indices)
            # check finish
            if (old_centroids == self.centroids).all():
                break

        return cluster_indices
    
    # assign centroid/ cluster to each data point:
    def assign_clusters(self, X):
        distances = jnp.linalg.norm(X[:, None] - self.centroids, axis=-1)
        cluster_indices = jnp.argmin(distances, axis=1)
        
        return cluster_indices

    # update the centroids:    
    def update_centroids(self, X, cluster_indices):
        updated_centroids = jnp.array([jnp.mean(X[cluster_indices == i], axis=0)\
                               for i in range(self.n_clusters)])
        
        return updated_centroids

