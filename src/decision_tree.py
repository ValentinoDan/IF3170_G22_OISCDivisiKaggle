import numpy as np


class DecisionTree:
    def __init__(
        self,
        criterion="id3",
        max_depth=None,
        min_samples_split=2,
        min_gain=0.0,
        ccp_alpha=0.0,
        random_state=42,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.root = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def plot(self, top_n=None, path=None):
        raise NotImplementedError

    def _build(self, X, y, depth):
        raise NotImplementedError

    def _split_score(self, X, y):
        raise NotImplementedError
