import numpy as np


class LogisticRegression:
    def __init__(
        self,
        penalty="l2",
        alpha=0.01,
        l1_ratio=0.5,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
        random_state=42,
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.theta = None
        self.loss_history = []

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def _sigmoid(self, z):
        raise NotImplementedError

    def _log_loss(self, X, y):
        raise NotImplementedError

    def _gradient(self, X, y):
        raise NotImplementedError
