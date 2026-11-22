from numbers import Integral
import numpy as np

class KNN:
    """KNN dengan jarak Euclidean dan suara mayoritas"""

    def __init__(self, n_neighbors=5):
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, Integral) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Simpan data latih dan labelnya"""
        X = np.array(X, dtype=float, order="C", copy=True)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array with at least one feature")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must contain one class label per row of X")
        if self.n_neighbors > len(X):
            raise ValueError("n_neighbors cannot exceed the number of training rows")

        self.classes_, self._label_indices = np.unique(y, return_inverse=True)
        self._training_rows = X
        return self

    def predict(self, X):
        """Jarak sama ikut urutan data latih, suara seri ambil label terkecil"""
        if not hasattr(self, "_training_rows"):
            raise ValueError("KNN must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._training_rows.shape[1] or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D array with the fitted number of features")

        predictions = np.empty(len(X), dtype=self.classes_.dtype)
        for index, query in enumerate(X):
            # Pakai kuadrat jarak, urutannya tetap sama
            offsets = self._training_rows - query
            distances = np.einsum("ij,ij->i", offsets, offsets)
            neighbors = np.argsort(distances, kind="stable")[:self.n_neighbors]
            votes = np.bincount(self._label_indices[neighbors], minlength=len(self.classes_))
            predictions[index] = self.classes_[votes.argmax()]
        return predictions