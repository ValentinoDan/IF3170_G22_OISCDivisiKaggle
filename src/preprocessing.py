import numpy as np
import pandas as pd


def load_data(train_path, test_path, target="Target"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def handle_missing(train, test):
    raise NotImplementedError


def handle_duplicates(train):
    raise NotImplementedError


def encode_categorical(train, test):
    raise NotImplementedError


def scale_numeric(train, test):
    raise NotImplementedError


def handle_imbalance(X, y, strategy="smote", random_state=42):
    raise NotImplementedError


def select_features(X, y, k=None):
    raise NotImplementedError


def reduce_dimensions(X_train, X_test, n_components=None):
    raise NotImplementedError


def split_features_target(df, target="Target"):
    raise NotImplementedError
