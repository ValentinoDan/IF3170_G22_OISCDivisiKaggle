import numpy as np
import pandas as pd

"""
Kelas untuk mengubah fitur numerik kontinu menjadi kategori interval berdasarkan distribusi data latih agar dapat diproses menggunakan tabel frekuensi diskrit
"""
class FeatureDiscretizer:

    # Setup awal untuk diskritisasi
    def __init__(self, n_bins=5, strategy="quantile"):
        if n_bins is not None and (not isinstance(n_bins, int) or n_bins < 2):
            raise ValueError(f"n_bins harus bilangan bulat >= 2, diterima: {n_bins}")
        if strategy not in ("quantile", "uniform"):
            raise ValueError("strategy harus 'quantile' atau 'uniform'")
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_ = {}

    # Melakukan fit untuk data
    def fit(self, X, numeric_cols):
        self.bin_edges_ = {}
        df = pd.DataFrame(X).copy()
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) == 0 or series.nunique() <= self.n_bins:
                self.bin_edges_[col] = None
                continue

            if self.strategy == "quantile":
                quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
                edges = np.unique(np.quantile(series.to_numpy(dtype=float), quantiles))
            else:
                edges = np.linspace(float(series.min()), float(series.max()), self.n_bins + 1)

            if len(edges) <= 2:
                self.bin_edges_[col] = None
            else:
                edges = edges.astype(float)
                edges[0] = -np.inf
                edges[-1] = np.inf
                self.bin_edges_[col] = edges
        return self

    # Melakukan transform untuk data
    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col, edges in self.bin_edges_.items():
            if col not in df.columns or edges is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            binned = pd.cut(vals, bins=edges, labels=False, include_lowest=True)
            df[col] = binned.astype("object")
            df.loc[vals.isna(), col] = np.nan
        return df

    # Fungsi gabungan fit + transform
    def fit_transform(self, X, numeric_cols):
        return self.fit(X, numeric_cols).transform(X)

"""
Model probabilitas kondisional P(a_i | v_j) berbasis frekuensi untuk atribut kategorikal/diskrit
"""
class CategoricalLikelihood:

    # Fungsi konstruktor
    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError(f"alpha harus >= 0, diterima: {alpha}")
        self.alpha = float(alpha)
        self.classes_ = None
        self.features_ = []
        self.categories_ = {}
        self.class_feature_Totals_ = {}
        self.frequency_table_ = {}
        self.cond_probs_ = {}
        self.unseen_prob_ = {}

    # Fungsi untuk fitting data
    def fit(self, X_df, y, classes, sample_weight=None):
        self.classes_ = np.asarray(classes)
        self.features_ = list(X_df.columns)
        self.categories_ = {}
        self.class_feature_Totals_ = {}
        self.frequency_table_ = {}
        self.cond_probs_ = {}
        self.unseen_prob_ = {}

        y_arr = np.asarray(y)
        weights = (
            np.ones(len(y_arr), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )

        for col in self.features_:
            col_series = X_df[col]
            valid_mask = ~pd.isna(col_series)
            unique_vals = pd.unique(col_series[valid_mask])
            try:
                unique_vals = sorted(unique_vals, key=lambda x: (str(type(x)), x))
            except TypeError:
                unique_vals = list(unique_vals)

            self.categories_[col] = list(unique_vals)
            n_categories = max(len(unique_vals), 1)

            self.frequency_table_[col] = {}
            self.cond_probs_[col] = {}
            self.class_feature_Totals_[col] = {}
            self.unseen_prob_[col] = {}

            for cls in self.classes_:
                cls_mask = (y_arr == cls) & valid_mask.to_numpy()
                cls_series = col_series[cls_mask]
                cls_weights = weights[cls_mask]
                total_cls_weight = float(np.sum(cls_weights))
                self.class_feature_Totals_[col][cls] = total_cls_weight

                freq_map = {}
                prob_map = {}
                denom = total_cls_weight + self.alpha * n_categories

                for val in unique_vals:
                    val_mask = (cls_series == val).to_numpy()
                    count = float(np.sum(cls_weights[val_mask]))
                    freq_map[val] = count
                    if denom > 0:
                        prob_map[val] = (count + self.alpha) / denom
                    else:
                        prob_map[val] = 1.0 / n_categories

                self.frequency_table_[col][cls] = freq_map
                self.cond_probs_[col][cls] = prob_map
                if denom > 0:
                    self.unseen_prob_[col][cls] = (
                        self.alpha / denom if self.alpha > 0 else 0.0
                    )
                else:
                    self.unseen_prob_[col][cls] = 1.0 / n_categories

        return self

    # Menghitung matriks proyeksi probabilitas secara langsung
    def likelihood_matrix(self, X_df, feature_weights=None):
        n_samples = len(X_df)
        n_classes = len(self.classes_)
        prod_prob = np.ones((n_samples, n_classes), dtype=float)

        if not self.features_:
            return prod_prob

        for idx, col in enumerate(self.features_):
            w = 1.0 if feature_weights is None else float(feature_weights[idx])
            col_vals = X_df[col]
            is_na = pd.isna(col_vals).to_numpy()

            for c_idx, cls in enumerate(self.classes_):
                prob_map = self.cond_probs_[col][cls]
                default_p = self.unseen_prob_[col][cls]
                mapped = col_vals.map(prob_map).to_numpy(dtype=float, copy=True)
                nan_from_unseen = np.isnan(mapped) & (~is_na)
                mapped[nan_from_unseen] = default_p

                # Mengubah entri menjadi 1.0 apabila NaN
                mapped[is_na] = 1.0
                if w != 1.0:
                    mapped = np.power(np.clip(mapped, 0.0, None), w)
                prod_prob[:, c_idx] *= mapped

        return prod_prob

    # Menghitung likelihood matriks dalam ruang logaritmik untuk mencegah underflowing floating point number
    def log_likelihood_matrix(self, X_df, feature_weights=None, eps=1e-15):
        n_samples = len(X_df)
        n_classes = len(self.classes_)
        log_prob = np.zeros((n_samples, n_classes), dtype=float)

        if not self.features_:
            return log_prob

        for idx, col in enumerate(self.features_):
            w = 1.0 if feature_weights is None else float(feature_weights[idx])
            col_vals = X_df[col]
            is_na = pd.isna(col_vals).to_numpy()

            for c_idx, cls in enumerate(self.classes_):
                prob_map = self.cond_probs_[col][cls]
                default_p = max(self.unseen_prob_[col][cls], eps)
                mapped = col_vals.map(prob_map).to_numpy(dtype=float, copy=True)
                nan_from_unseen = np.isnan(mapped) & (~is_na)
                mapped[nan_from_unseen] = default_p
                mapped = np.clip(mapped, eps, None)
                col_log_p = np.log(mapped)
                col_log_p[is_na] = 0.0
                log_prob[:, c_idx] += w * col_log_p

        return log_prob

"""
Model probabilitas kondisional berbasis distribusi Gaussian untuk atribut numerik kontinu
"""
class GaussianLikelihood:

    # Fungsi konstruktor
    def __init__(self, var_smoothing=1e-9):
        if var_smoothing < 0:
            raise ValueError(f"var_smoothing harus >= 0, diterima: {var_smoothing}")
        self.var_smoothing = float(var_smoothing)
        self.classes_ = None
        self.features_ = []
        self.means_ = None
        self.vars_ = None
        self.epsilon_ = 0.0

    # Fitting dataset
    def fit(self, X_df, y, classes, sample_weight=None):
        self.classes_ = np.asarray(classes)
        self.features_ = list(X_df.columns)
        n_classes = len(self.classes_)
        n_features = len(self.features_)

        if n_features == 0:
            self.means_ = np.empty((n_classes, 0), dtype=float)
            self.vars_ = np.empty((n_classes, 0), dtype=float)
            return self

        X_num = X_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        y_arr = np.asarray(y)
        weights = (
            np.ones(len(y_arr), dtype=float)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )

        global_vars = np.array(
            [
                float(np.var(X_num[~np.isnan(X_num[:, i]), i]))
                if np.any(~np.isnan(X_num[:, i]))
                else 0.0
                for i in range(n_features)
            ],
            dtype=float,
        )
        max_var = float(np.max(global_vars)) if len(global_vars) > 0 else 1.0
        self.epsilon_ = self.var_smoothing * (max_var if max_var > 0 else 1.0)

        self.means_ = np.zeros((n_classes, n_features), dtype=float)
        self.vars_ = np.zeros((n_classes, n_features), dtype=float)

        for c_idx, cls in enumerate(self.classes_):
            cls_mask = y_arr == cls
            X_cls = X_num[cls_mask]
            w_cls = weights[cls_mask]

            for f_idx in range(n_features):
                col_vals = X_cls[:, f_idx]
                valid = ~np.isnan(col_vals)
                if not np.any(valid):
                    self.means_[c_idx, f_idx] = 0.0
                    self.vars_[c_idx, f_idx] = 1.0 + self.epsilon_
                    continue

                v_vals = col_vals[valid]
                v_w = w_cls[valid]
                w_sum = float(np.sum(v_w))
                if w_sum <= 0:
                    mean_val = float(np.mean(v_vals))
                    var_val = float(np.var(v_vals))
                else:
                    mean_val = float(np.sum(v_w * v_vals) / w_sum)
                    var_val = float(np.sum(v_w * ((v_vals - mean_val) ** 2)) / w_sum)

                self.means_[c_idx, f_idx] = mean_val
                self.vars_[c_idx, f_idx] = var_val + self.epsilon_

        return self

    # Menghitung matriks sum_i w_i * log P(a_i | v_j) untuk fitur numerik kontinu
    def log_likelihood_matrix(self, X_df, feature_weights=None):
        n_samples = len(X_df)
        n_classes = len(self.classes_)
        if len(self.features_) == 0:
            return np.zeros((n_samples, n_classes), dtype=float)

        X_num = X_df[self.features_].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        is_na = np.isnan(X_num)
        X_filled = np.where(is_na, 0.0, X_num)

        weights = (
            np.ones(len(self.features_), dtype=float)
            if feature_weights is None
            else np.asarray(feature_weights, dtype=float)
        )

        log_prob = np.zeros((n_samples, n_classes), dtype=float)
        for c_idx in range(n_classes):
            mean = self.means_[c_idx]
            var = np.maximum(self.vars_[c_idx], 1e-15)
            diff_sq = (X_filled - mean) ** 2
            comp_log_pdf = -0.5 * (np.log(2.0 * np.pi * var) + (diff_sq / var))
            comp_log_pdf = np.where(is_na, 0.0, comp_log_pdf)
            log_prob[:, c_idx] = np.sum(comp_log_pdf * weights, axis=1)

        return log_prob

    # Fungsi warapper untuk menghitung likelihood matriks
    def likelihood_matrix(self, X_df, feature_weights=None):
        """
        Menghitung matriks prod_i P(a_i | v_j) dalam ruang probabilitas langsung.
        """
        return np.exp(self.log_likelihood_matrix(X_df, feature_weights=feature_weights))
