import itertools
import numpy as np
import pandas as pd

try:
    from .. import utils
except Exception:
    try:
        import utils
    except Exception:
        utils = None

# Menghitung akurasi
def _compute_accuracy(y_true, y_pred):
    if utils is not None and hasattr(utils, "accuracy"):
        return utils.accuracy(y_true, y_pred)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

# Menghitung bobot fitur w_i berbasis Mutual Information I(A_i; V) secara from scratch untuk metode Feature-Weighted Naive Bayes
def compute_mutual_information_weights(X_df, y, n_bins=10, min_weight=0.05):
    df = pd.DataFrame(X_df).copy()
    y_arr = np.asarray(y)
    n_samples = len(y_arr)
    n_features = df.shape[1]

    if n_features == 0 or n_samples == 0:
        return np.ones(n_features, dtype=float)

    classes, y_counts = np.unique(y_arr, return_counts=True)
    p_y = {cls: count / n_samples for cls, count in zip(classes, y_counts)}

    mi_scores = np.zeros(n_features, dtype=float)

    for idx, col in enumerate(df.columns):
        series = df[col]
        valid_mask = ~pd.isna(series)
        if not np.any(valid_mask):
            mi_scores[idx] = 0.0
            continue

        s_valid = series[valid_mask]
        y_valid = y_arr[valid_mask.to_numpy()]
        n_valid = len(y_valid)

        if pd.api.types.is_numeric_dtype(s_valid) and s_valid.nunique() > n_bins:
            try:
                discretized = pd.qcut(
                    s_valid.to_numpy(dtype=float), q=n_bins, labels=False, duplicates="drop"
                )
                vals = pd.Series(discretized, index=s_valid.index)
            except ValueError:
                vals = s_valid
        else:
            vals = s_valid

        unique_a, a_counts = np.unique(vals.to_numpy(dtype=str), return_counts=True)
        a_str = vals.to_numpy(dtype=str)
        p_a = {a_val: cnt / n_valid for a_val, cnt in zip(unique_a, a_counts)}

        mi = 0.0
        for a_val in unique_a:
            a_mask = a_str == a_val
            pa = p_a[a_val]
            if pa <= 0:
                continue
            for cls in classes:
                joint_count = np.sum(a_mask & (y_valid == cls))
                if joint_count <= 0:
                    continue
                p_av = joint_count / n_valid
                pv = p_y[cls]
                if pv > 0:
                    mi += p_av * np.log(p_av / (pa * pv))

        mi_scores[idx] = max(0.0, float(mi))

    total_mi = float(np.sum(mi_scores))
    if total_mi <= 1e-12:
        return np.ones(n_features, dtype=float)

    # Normalisasi bobot sehingga rata-rata bobot = 1.0 (skala log-likelihood tetap terjaga)
    weights = (mi_scores / total_mi) * n_features
    weights = np.maximum(weights, min_weight)
    return weights

# Menghitung ambang batas keputusan (decision threshold) optimal berdasarkan Minimum Expected Cost Bayes Decision Theory
def compute_cost_sensitive_threshold(cost_fp=2.0, cost_fn=1.0):
    if cost_fp <= 0 or cost_fn <= 0:
        raise ValueError("cost_fp dan cost_fn harus bernilai positif (> 0)")
    return float(cost_fp) / float(cost_fp + cost_fn)

# Menghitung metrik evaluasi klasifikasi beserta analisis biaya
def evaluate_classification_metrics(y_true, y_pred, pos_label=1, cost_fp=2.0, cost_fn=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
    tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))
    fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))

    acc = _compute_accuracy(y_true, y_pred)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    total_cost = float(cost_fp * fp + cost_fn * fn)
    avg_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_cost": total_cost,
        "average_cost": avg_cost,
    }

# Melakukan pencarian kombinasi hyperparameter terbaik untuk NaiveBayes (mendukung optimasi akurasi, F1-score, maupun minimisasi biaya cost-sensitive)
def tune_naive_bayes(model_cls, X_train, y_train, X_val, y_val, param_grid=None, metric="f1", cost_fp=2.0, cost_fn=1.0):

    if param_grid is None:
        param_grid = {
            "mode": ["auto", "gaussian", "discretized"],
            "var_smoothing": [1e-9, 1e-5, 1e-2, 1e-1],
            "alpha": [0.5, 1.0, 2.0],
            "use_feature_weights": [False, True],
        }

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))

    best_model = None
    best_params = None
    best_score = -np.inf if metric != "cost" else np.inf
    history = []

    for values in combinations:
        params = dict(zip(keys, values))
        model = model_cls(**params)
        model.fit(X_train, y_train)

        if metric == "cost":
            opt_thresh = compute_cost_sensitive_threshold(cost_fp=cost_fp, cost_fn=cost_fn)
            preds = model.predict(X_val, threshold=opt_thresh)
        else:
            preds = model.predict(X_val)

        metrics = evaluate_classification_metrics(
            y_val, preds, cost_fp=cost_fp, cost_fn=cost_fn
        )
        record = {**params, **metrics}
        history.append(record)

        if metric == "cost":
            score = metrics["total_cost"]
            is_better = score < best_score
        else:
            score = metrics.get(metric, metrics["accuracy"])
            is_better = score > best_score

        if is_better:
            best_score = score
            best_params = params
            best_model = model

    history_df = pd.DataFrame(history)
    return best_model, best_params, history_df
