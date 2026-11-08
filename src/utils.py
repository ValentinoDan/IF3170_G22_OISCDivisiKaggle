import pickle

import numpy as np

ACCENT = "#a8052b"
TREE_ACCENT = "#1d8a51"


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    raise NotImplementedError


def precision_recall_f1(y_true, y_pred, average="binary"):
    raise NotImplementedError


def plot_metric_comparison(results, path=None):
    raise NotImplementedError
