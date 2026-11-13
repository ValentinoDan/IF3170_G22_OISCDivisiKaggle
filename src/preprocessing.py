from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

TARGET = "is_canceled"
ID_COL = "id"
RANDOM_STATE = 42

def find_data_dir(root=".", candidates=("dataset", "data")):
    root = Path(root)
    for name in candidates:
        path = root / name
        if (path / "train.csv").exists() and (path / "test.csv").exists():
            return path
    raise FileNotFoundError("train.csv dan test.csv tidak ditemukan pada folder: " + ", ".join(str(root / name) for name in candidates))

def load_data(data_dir=None):
    if data_dir is None:
        data_dir = find_data_dir()
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test

def summarize(df):
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
            "sample": [df[col].dropna().iloc[0] if df[col].notna().any() else np.nan for col in df.columns],
        }
    )

def clean_data(train, test, target=TARGET):
    report = {}

    report["duplicates_train"] = int(train.duplicated().sum())
    train = train.drop_duplicates().copy()

    report["missing_target"] = int(train[target].isna().sum())
    train = train.dropna(subset=[target]).copy()
    train[target] = train[target].astype(int)

    zero_adults_train = int((train["num_adults"] == 0).sum())
    zero_adults_test = int((test["num_adults"] == 0).sum())
    report["zero_adults_train"] = zero_adults_train
    report["zero_adults_test"] = zero_adults_test
    for df in (train, test):
        df.loc[df["num_adults"] == 0, "num_adults"] = np.nan

    zero_rate_train = int(((train["nightly_rate_idr"] == 0) & (train["booking_channel"] != "Complimentary")).sum())
    zero_rate_test = int(((test["nightly_rate_idr"] == 0) & (test["booking_channel"] != "Complimentary")).sum())
    report["zero_rate_non_complimentary_train"] = zero_rate_train
    report["zero_rate_non_complimentary_test"] = zero_rate_test
    for df in (train, test):
        mask = (df["nightly_rate_idr"] == 0) & (df["booking_channel"] != "Complimentary")
        df.loc[mask, "nightly_rate_idr"] = np.nan

    report["zero_nights_train"] = int(((train["weekend_nights"] == 0) & (train["weekday_nights"] == 0)).sum())
    report["zero_nights_test"] = int(((test["weekend_nights"] == 0) & (test["weekday_nights"] == 0)).sum())

    return train, test, report

def split_features_target(df, target=TARGET, id_col=ID_COL):
    drop_cols = [col for col in (target, id_col) if col in df.columns]
    X = df.drop(columns=drop_cols).copy()
    y = df[target].copy() if target in df.columns else None
    ids = df[id_col].copy() if id_col in df.columns else None
    return X, y, ids

def get_feature_types(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in X.columns if col not in numeric]
    return numeric, categorical

def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def fit_transform_preprocessor(preprocessor, X_train, X_val, X_test):
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame(X_train_prep, columns=feature_names, index=X_train.index),
        pd.DataFrame(X_val_prep, columns=feature_names, index=X_val.index),
        pd.DataFrame(X_test_prep, columns=feature_names, index=X_test.index),
        list(feature_names),
    )

def class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))

def resample_train(X, y, strategy="oversample", random_state=RANDOM_STATE):
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    counts = y.value_counts()

    if strategy == "oversample":
        target_count = int(counts.max())
        parts_X, parts_y = [], []
        for label, count in counts.items():
            X_class = X[y == label]
            y_class = y[y == label]
            if count < target_count:
                X_class, y_class = resample(
                    X_class,
                    y_class,
                    replace=True,
                    n_samples=target_count,
                    random_state=random_state,
                )
            parts_X.append(X_class)
            parts_y.append(y_class)
        X_res = pd.concat(parts_X).reset_index(drop=True)
        y_res = pd.concat(parts_y).reset_index(drop=True)
        order = np.random.RandomState(random_state).permutation(len(X_res))
        return X_res.iloc[order].reset_index(drop=True), y_res.iloc[order].reset_index(drop=True)
    
    if strategy == "undersample":
        target_count = int(counts.min())
        parts_X, parts_y = [], []
        for label in counts.index:
            X_class = X[y == label]
            y_class = y[y == label]
            if len(X_class) > target_count:
                X_class, y_class = resample(
                    X_class,
                    y_class,
                    replace=False,
                    n_samples=target_count,
                    random_state=random_state,
                )
            parts_X.append(X_class)
            parts_y.append(y_class)

        X_res = pd.concat(parts_X).reset_index(drop=True)
        y_res = pd.concat(parts_y).reset_index(drop=True)
        order = np.random.RandomState(random_state).permutation(len(X_res))
        return X_res.iloc[order].reset_index(drop=True), y_res.iloc[order].reset_index(drop=True)
    raise ValueError("strategy harus 'oversample' atau 'undersample'")

def select_features(X_train, y_train, X_val=None, X_test=None, k=None, variance_threshold=0.0):
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(X_train)
    kept = np.asarray(X_train.columns)[selector.get_support()]

    X_train = X_train[kept]
    X_val = X_val[kept] if X_val is not None else None
    X_test = X_test[kept] if X_test is not None else None

    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    mi_scores = pd.Series(mi, index=X_train.columns, name="mutual_info").sort_values(ascending=False)

    if k is not None:
        selected = mi_scores.head(k).index.tolist()
    else:
        selected = mi_scores[mi_scores > 0].index.tolist()
        if not selected:
            selected = mi_scores.head(min(10, len(mi_scores))).index.tolist()

    X_train = X_train[selected]
    X_val = X_val[selected] if X_val is not None else None
    X_test = X_test[selected] if X_test is not None else None
    return X_train, X_val, X_test, selected, mi_scores

def fit_pca(X_train, X_val, X_test, n_components=None, variance=0.95):
    pca = PCA(n_components=n_components if n_components is not None else variance, random_state=RANDOM_STATE)
    Z_train = pca.fit_transform(X_train)
    Z_val = pca.transform(X_val)
    Z_test = pca.transform(X_test)
    return pca, Z_train, Z_val, Z_test

def save_processed_data(processed_dir, model_dir, preprocessor, pca, selected_features, X_train, y_train, X_val, y_val, X_test,
                        test_ids, Z_train=None, Z_val=None, Z_test=None):
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    joblib.dump(pca, model_dir / "pca.joblib")
    pd.Series(selected_features, name="feature").to_csv(processed_dir / "selected_features.csv", index=False)
    test_ids.to_frame("id").to_csv(processed_dir / "test_ids.csv", index=False)

    arrays = {
        "X_train": np.asarray(X_train),
        "y_train": np.asarray(y_train),
        "X_val": np.asarray(X_val),
        "y_val": np.asarray(y_val),
        "X_test": np.asarray(X_test),
    }
    if Z_train is not None:
        arrays.update(
            {
                "Z_train": np.asarray(Z_train),
                "Z_val": np.asarray(Z_val),
                "Z_test": np.asarray(Z_test),
            }
        )
    np.savez_compressed(processed_dir / "data.npz", **arrays)