import numpy as np


class SVM:
    def __init__(
        self,
        kernel="linear",
        C=1.0,
        degree=3,
        gamma="scale",
        coef0=0.0,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
        random_state=42,
    ):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.support_vectors = None
        self.dual_coef = None
        self.bias = 0.0

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError

    def _kernel(self, A, B):
        raise NotImplementedError
