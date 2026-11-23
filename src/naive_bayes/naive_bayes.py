from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from .distributions import CategoricalLikelihood, FeatureDiscretizer, GaussianLikelihood
from .optimizer import (
    compute_cost_sensitive_threshold,
    compute_mutual_information_weights,
    evaluate_classification_metrics,
    tune_naive_bayes,
)
from .visualizer import export_probability_tables, plot_bayesian_network

try:
    from .. import preprocessing, utils
except Exception:
    try:
        import preprocessing, utils
    except Exception:
        preprocessing = None
        utils = None

# Memanfaatkan fungsi preprocessing.get_feature_types(X) apabila modul preprocessing tersedia, atau fallback identik dengan implementasi di preprocessing.py
def _get_feature_types(X_df):
    if preprocessing is not None and hasattr(preprocessing, "get_feature_types"):
        return preprocessing.get_feature_types(X_df)
    numeric = X_df.select_dtypes(include=np.number).columns.tolist()
    categorical = [col for col in X_df.columns if col not in numeric]
    return numeric, categorical

# Memanfaatkan fungsi preprocessing.class_weights(y) apabila tersedia untuk menghitung bobot kelas seimbang.
def _compute_balanced_priors(y):
    if preprocessing is not None and hasattr(preprocessing, "class_weights"):
        weights_dict = preprocessing.class_weights(y)
        total = sum(weights_dict.values())
        return {k: v / total for k, v in weights_dict.items()}
    classes, counts = np.unique(y, return_counts=True)
    inv = 1.0 / np.maximum(counts, 1)
    inv /= np.sum(inv)
    return dict(zip(classes.tolist(), inv.tolist()))

"""
Implementasi Algoritma Bayesian Network (Naive Bayes) from scratch
"""
class NaiveBayes:
    VALID_MODES = ("auto", "gaussian", "categorical", "mixed", "discretized")

    def __init__(self, mode="auto", var_smoothing=1e-9, alpha=1.0, n_bins=5, discretize_strategy="quantile", priors=None, use_feature_weights=False, decision_threshold=None, max_categorical_cardinality=15):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode harus salah satu dari {self.VALID_MODES}, diterima: {mode}")
        if var_smoothing < 0:
            raise ValueError(f"var_smoothing harus >= 0, diterima: {var_smoothing}")
        if alpha < 0:
            raise ValueError(f"alpha harus >= 0, diterima: {alpha}")

        self.mode = mode
        self.var_smoothing = float(var_smoothing)
        self.alpha = float(alpha)
        self.n_bins = n_bins
        self.discretize_strategy = discretize_strategy
        self.priors = priors
        self.use_feature_weights = bool(use_feature_weights)
        self.decision_threshold = decision_threshold
        self.max_categorical_cardinality = int(max_categorical_cardinality)

        self.classes_ = None
        self.classes = None
        self.class_counts_ = None
        self.class_priors_ = None
        self.class_priors = None
        self.feature_names_ = []
        self.numeric_features_ = []
        self.categorical_features_ = []
        self.discretizer_ = None
        self.categorical_model_ = None
        self.gaussian_model_ = None
        self.frequency_table_ = {}
        self.cond_probs_ = {}
        self.means_ = None
        self.vars_ = None
        self.means = None
        self.vars = None
        self.feature_weights_ = None
        self.feature_weight_map_ = {}

    # Fungsi helper untuk mengubah ke dataframe
    def _to_dataframe(self, X, is_fit=False):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if is_fit:
                self.feature_names_ = [str(col) for col in df.columns]
            df.columns = [str(col) for col in df.columns]
            if not is_fit and self.feature_names_:
                missing_cols = [c for c in self.feature_names_ if c not in df.columns]
                for c in missing_cols:
                    df[c] = np.nan
                df = df[self.feature_names_]
            return df

        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if is_fit or not self.feature_names_:
            cols = [f"feature_{i}" for i in range(arr.shape[1])]
            if is_fit:
                self.feature_names_ = cols
        else:
            if arr.shape[1] != len(self.feature_names_):
                raise ValueError(
                    f"Jumlah fitur input ({arr.shape[1]}) tidak sesuai dengan "
                    f"jumlah fitur saat training ({len(self.feature_names_)})"
                )
            cols = self.feature_names_
        return pd.DataFrame(arr, columns=cols)

    # Mengecek mode untuk tiap2 fitur
    def _partition_features(self, X_df):
        numeric_cols, categorical_cols = _get_feature_types(X_df)

        if self.mode == "categorical":
            return [], list(X_df.columns)
        if self.mode == "gaussian":
            return list(X_df.columns), []
        if self.mode == "discretized":
            return list(numeric_cols), list(categorical_cols)
        if self.mode == "mixed":
            true_num = []
            disc_cat = list(categorical_cols)
            for col in numeric_cols:
                n_uniq = X_df[col].dropna().nunique()
                if n_uniq <= self.max_categorical_cardinality:
                    disc_cat.append(col)
                else:
                    true_num.append(col)
            return true_num, disc_cat

        return list(numeric_cols), list(categorical_cols)

    # Melatih model naive bayes
    def fit(self, X, y, sample_weight=None):
        X_df = self._to_dataframe(X, is_fit=True)
        y_arr = np.asarray(y)

        if len(X_df) != len(y_arr):
            raise ValueError(
                f"Panjang X ({len(X_df)}) dan y ({len(y_arr)}) harus sama."
            )

        self.classes_ = np.unique(y_arr)
        self.classes = self.classes_
        if len(self.classes_) < 2:
            raise ValueError(
                f"NaiveBayes membutuhkan minimal 2 kelas berbeda, ditemukan: {self.classes_}"
            )

        weights = (
            np.ones(len(y_arr), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        total_weight = float(np.sum(weights))

        self.class_counts_ = {}
        self.class_priors_ = {}
        for cls in self.classes_:
            cls_w = float(np.sum(weights[y_arr == cls]))
            self.class_counts_[cls] = cls_w
            self.class_priors_[cls] = cls_w / total_weight if total_weight > 0 else 1.0 / len(self.classes_)

        if self.priors == "balanced":
            self.class_priors_ = _compute_balanced_priors(y_arr)
        elif isinstance(self.priors, dict):
            total_p = sum(self.priors.values())
            self.class_priors_ = {
                cls: float(self.priors[cls]) / total_p for cls in self.classes_
            }
        elif self.priors is not None:
            prior_arr = np.asarray(self.priors, dtype=float)
            if len(prior_arr) != len(self.classes_):
                raise ValueError("Panjang priors harus sama dengan jumlah kelas.")
            prior_arr = prior_arr / np.sum(prior_arr)
            self.class_priors_ = {
                cls: float(prior_arr[i]) for i, cls in enumerate(self.classes_)
            }

        self.class_priors = np.array(
            [self.class_priors_[cls] for cls in self.classes_], dtype=float
        )

        num_cols, cat_cols = self._partition_features(X_df)

        if self.mode == "discretized" and len(num_cols) > 0:
            self.discretizer_ = FeatureDiscretizer(
                n_bins=self.n_bins, strategy=self.discretize_strategy
            )
            X_proc = self.discretizer_.fit_transform(X_df, num_cols)
            self.numeric_features_ = []
            self.categorical_features_ = list(X_df.columns)
        else:
            self.discretizer_ = None
            X_proc = X_df
            self.numeric_features_ = num_cols
            self.categorical_features_ = cat_cols

        if self.use_feature_weights:
            self.feature_weights_ = compute_mutual_information_weights(X_proc, y_arr)
        else:
            self.feature_weights_ = np.ones(len(self.feature_names_), dtype=float)

        self.feature_weight_map_ = {
            feat: float(self.feature_weights_[i])
            for i, feat in enumerate(self.feature_names_)
        }

        if len(self.categorical_features_) > 0:
            self.categorical_model_ = CategoricalLikelihood(alpha=self.alpha)
            self.categorical_model_.fit(
                X_proc[self.categorical_features_],
                y_arr,
                self.classes_,
                sample_weight=weights,
            )
            self.frequency_table_ = self.categorical_model_.frequency_table_
            self.cond_probs_ = self.categorical_model_.cond_probs_
        else:
            self.categorical_model_ = None
            self.frequency_table_ = {}
            self.cond_probs_ = {}

        if len(self.numeric_features_) > 0:
            self.gaussian_model_ = GaussianLikelihood(var_smoothing=self.var_smoothing)
            self.gaussian_model_.fit(
                X_proc[self.numeric_features_],
                y_arr,
                self.classes_,
                sample_weight=weights,
            )
            self.means_ = self.gaussian_model_.means_
            self.vars_ = self.gaussian_model_.vars_
        else:
            self.gaussian_model_ = None
            self.means_ = None
            self.vars_ = None

        self.means = self.means_
        self.vars = self.vars_
        return self

    # Mengecek apakah model sudah di fitted
    def _check_is_fitted(self):
        if self.classes_ is None or self.class_priors is None:
            raise ValueError("Model NaiveBayes belum di-fit. Panggil fit(X, y) terlebih dahulu.")

    # Menyiapkan input
    def _prepare_input(self, X):
        self._check_is_fitted()
        X_df = self._to_dataframe(X, is_fit=False)
        if self.discretizer_ is not None:
            X_df = self.discretizer_.transform(X_df)
        return X_df

    # Menghitung log-likelihood gabungan sum_i w_i * log P(a_i | v_j) untuk setiap sampel dan setiap kelas v_j.
    def _log_likelihood(self, X):
        X_df = self._prepare_input(X)
        n_samples = len(X_df)
        n_classes = len(self.classes_)
        total_log_likelihood = np.zeros((n_samples, n_classes), dtype=float)

        if self.categorical_model_ is not None and len(self.categorical_features_) > 0:
            cat_weights = [
                self.feature_weight_map_.get(col, 1.0)
                for col in self.categorical_features_
            ]
            total_log_likelihood += self.categorical_model_.log_likelihood_matrix(
                X_df[self.categorical_features_], feature_weights=cat_weights
            )

        if self.gaussian_model_ is not None and len(self.numeric_features_) > 0:
            num_weights = [
                self.feature_weight_map_.get(col, 1.0)
                for col in self.numeric_features_
            ]
            total_log_likelihood += self.gaussian_model_.log_likelihood_matrix(
                X_df[self.numeric_features_], feature_weights=num_weights
            )

        return total_log_likelihood

    # Menghitung rumus klasifikasi Naive Bayes secara langsung
    def compute_unnormalized_posterior(self, X):
        X_df = self._prepare_input(X)
        n_samples = len(X_df)
        n_classes = len(self.classes_)
        joint_prob = np.tile(self.class_priors, (n_samples, 1))

        if self.categorical_model_ is not None and len(self.categorical_features_) > 0:
            cat_weights = [
                self.feature_weight_map_.get(col, 1.0)
                for col in self.categorical_features_
            ]
            joint_prob *= self.categorical_model_.likelihood_matrix(
                X_df[self.categorical_features_], feature_weights=cat_weights
            )

        if self.gaussian_model_ is not None and len(self.numeric_features_) > 0:
            num_weights = [
                self.feature_weight_map_.get(col, 1.0)
                for col in self.numeric_features_
            ]
            joint_prob *= self.gaussian_model_.likelihood_matrix(
                X_df[self.numeric_features_], feature_weights=num_weights
            )

        return joint_prob

    # Menghitung log posterior ternormalisasi log P(v_j | a_1, ..., a_n) menggunakan log-sum-exp trick untuk stabilitas numerik
    def predict_log_proba(self, X):
        log_priors = np.log(np.clip(self.class_priors, 1e-15, None))
        jll = self._log_likelihood(X) + log_priors
        max_jll = np.max(jll, axis=1, keepdims=True)
        log_prob_x = max_jll + np.log(np.sum(np.exp(jll - max_jll), axis=1, keepdims=True))
        return jll - log_prob_x

    # Menghitung probabilitas posterior ternormalisasi untuk setiap kelas v_j in V.
    def predict_proba(self, X):
        if self.alpha == 0.0 and self.gaussian_model_ is None:
            unnorm = self.compute_unnormalized_posterior(X)
            row_sums = np.sum(unnorm, axis=1, keepdims=True)
            uniform = np.full_like(unnorm, 1.0 / len(self.classes_))
            return np.divide(unnorm, row_sums, out=uniform, where=row_sums > 0)

        return np.exp(self.predict_log_proba(X))

    # Memprediksi kelas untuk data uji X yang mendukung Cost-Sensitive Classification apabila `threshold` atau pasangan (`cost_fp`, `cost_fn`) diberikan
    def predict(self, X, threshold=None, cost_fp=None, cost_fn=None):
        self._check_is_fitted()

        active_threshold = threshold
        if active_threshold is None and cost_fp is not None and cost_fn is not None:
            active_threshold = compute_cost_sensitive_threshold(
                cost_fp=cost_fp, cost_fn=cost_fn
            )
        if active_threshold is None:
            active_threshold = self.decision_threshold

        if active_threshold is not None and len(self.classes_) == 2:
            proba = self.predict_proba(X)
            pos_idx = 1
            neg_idx = 0
            return np.where(
                proba[:, pos_idx] >= float(active_threshold),
                self.classes_[pos_idx],
                self.classes_[neg_idx],
            )

        if self.alpha == 0.0 and self.gaussian_model_ is None:
            scores = self.compute_unnormalized_posterior(X)
        else:
            log_priors = np.log(np.clip(self.class_priors, 1e-15, None))
            scores = self._log_likelihood(X) + log_priors

        best_indices = np.argmax(scores, axis=1)
        return self.classes_[best_indices]

    # Menghitung akurasi model menggunakan fungsi utils.accuracy(y_true, y_pred)
    def score(self, X, y):
        y_pred = self.predict(X)
        if utils is not None and hasattr(utils, "accuracy"):
            return utils.accuracy(y, y_pred)
        return float(np.mean(np.asarray(y) == y_pred))

    # Mengevaluasi performa model pada data (X, y) mencakup akurasi, presisi, recall, F1-score, confusion matrix, dan total biaya (cost-sensitive).
    def evaluate(self, X, y, threshold=None, cost_fp=2.0, cost_fn=1.0):
        y_pred = self.predict(X, threshold=threshold)
        return evaluate_classification_metrics(
            y, y_pred, cost_fp=cost_fp, cost_fn=cost_fn
        )

    # Menghasilkan tabel Frequency dan Probability Model (P(v_j) dan P(a_i | v_j))
    def get_probability_tables(self, prefix_path=None):
        return export_probability_tables(self, prefix_path=prefix_path)

    # Menghasilkan visualisasi graf Bayesian Network / Naive Bayes dan menyimpannya menggunakan utils.save_figure
    def plot_network(self, path=None, top_n=12, target_name="is_canceled", figsize=(14, 7)):
        return plot_bayesian_network(
            self,
            path=path,
            top_n=top_n,
            target_name=target_name,
            figsize=figsize,
        )

    # Menyimpan model terlatih ke berkas menggunakan utils.save_model
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if utils is not None and hasattr(utils, "save_model"):
            utils.save_model(self, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    # Memuat model terlatih dari berkas menggunakan utils.load_model
    def load(cls, path):
        if utils is not None and hasattr(utils, "load_model"):
            return utils.load_model(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    # Menjalankan optimasi hyperparameter dan feature weighting
    def optimize(cls, X_train, y_train, X_val, y_val, param_grid=None, metric="f1", cost_fp=2.0, cost_fn=1.0):
        return tune_naive_bayes(
            cls,
            X_train,
            y_train,
            X_val,
            y_val,
            param_grid=param_grid,
            metric=metric,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )
