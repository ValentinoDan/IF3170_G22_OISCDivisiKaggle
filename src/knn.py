import numpy as np


class KNN:
    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def _distance(self, X):
        raise NotImplementedError
