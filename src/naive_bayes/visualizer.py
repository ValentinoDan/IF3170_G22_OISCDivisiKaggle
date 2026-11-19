from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

try:
    from .. import utils
except Exception:
    try:
        import utils
    except Exception:
        utils = None

CHART_COLOR = getattr(utils, "CHART", "#0000FF")

# Menyimpan figure
def _save_figure(fig, path):
    if utils is not None and hasattr(utils, "save_figure"):
        utils.save_figure(fig, path)
        return
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# Menyimpan table
def _save_table(df, path):
    if path is None:
        return
    if utils is not None and hasattr(utils, "save_table"):
        utils.save_table(df, path)
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

# Mengekstrak tabel Frequency & Probability Model. Jika prefix_path diberikan, tabel disimpan menggunakan utils.save_table.
def export_probability_tables(model, prefix_path=None):
    model._check_is_fitted()

    prior_rows = []
    for cls in model.classes_:
        prior_rows.append(
            {
                "class": cls,
                "frequency": model.class_counts_[cls],
                "prior_probability_P(v_j)": model.class_priors_[cls],
            }
        )
    prior_df = pd.DataFrame(prior_rows).set_index("class")

    cat_rows = []
    if model.categorical_model_ is not None:
        for feat in model.categorical_model_.features_:
            for val in model.categorical_model_.categories_.get(feat, []):
                row = {"feature": feat, "attribute_value": val}
                for cls in model.classes_:
                    freq = model.categorical_model_.frequency_table_[feat][cls].get(val, 0.0)
                    prob = model.categorical_model_.cond_probs_[feat][cls].get(val, 0.0)
                    row[f"freq({val}|{cls})"] = freq
                    row[f"P({val}|{cls})"] = prob
                cat_rows.append(row)
    categorical_df = pd.DataFrame(cat_rows)

    gauss_rows = []
    if model.gaussian_model_ is not None and len(model.gaussian_model_.features_) > 0:
        for f_idx, feat in enumerate(model.gaussian_model_.features_):
            row = {"feature": feat}
            for c_idx, cls in enumerate(model.classes_):
                row[f"mean_{cls}"] = float(model.gaussian_model_.means_[c_idx, f_idx])
                row[f"var_{cls}"] = float(model.gaussian_model_.vars_[c_idx, f_idx])
            gauss_rows.append(row)
    gaussian_df = pd.DataFrame(gauss_rows)

    if prefix_path is not None:
        base = Path(prefix_path)
        _save_table(prior_df, f"{base}_priors.csv")
        if not categorical_df.empty:
            _save_table(categorical_df, f"{base}_categorical_likelihoods.csv")
        if not gaussian_df.empty:
            _save_table(gaussian_df, f"{base}_gaussian_parameters.csv")

    return {
        "priors": prior_df,
        "categorical": categorical_df,
        "gaussian": gaussian_df,
    }

# Menghitung skor kekuatan pengaruh (discriminative power) tiap fitur terhadap variabel kelas V untuk memberi bobot visual pada graf Naive Bayes.
def _compute_feature_discriminative_scores(model):

    scores = {}
    classes = model.classes_

    if model.feature_weights_ is not None and len(model.feature_names_) == len(model.feature_weights_):
        for name, w in zip(model.feature_names_, model.feature_weights_):
            scores[name] = float(w)
        return scores

    if model.categorical_model_ is not None:
        for feat in model.categorical_model_.features_:
            vals = model.categorical_model_.categories_.get(feat, [])
            if len(classes) >= 2 and len(vals) > 0:
                p0 = np.array(
                    [model.categorical_model_.cond_probs_[feat][classes[0]].get(v, 0.0) for v in vals]
                )
                p1 = np.array(
                    [model.categorical_model_.cond_probs_[feat][classes[-1]].get(v, 0.0) for v in vals]
                )
                scores[feat] = float(0.5 * np.sum(np.abs(p0 - p1)))
            else:
                scores[feat] = 1.0

    if model.gaussian_model_ is not None:
        for f_idx, feat in enumerate(model.gaussian_model_.features_):
            if len(classes) >= 2:
                m0 = model.gaussian_model_.means_[0, f_idx]
                m1 = model.gaussian_model_.means_[-1, f_idx]
                v0 = model.gaussian_model_.vars_[0, f_idx]
                v1 = model.gaussian_model_.vars_[-1, f_idx]
                pooled_std = np.sqrt(0.5 * (v0 + v1) + 1e-12)
                scores[feat] = float(abs(m1 - m0) / pooled_std)
            else:
                scores[feat] = 1.0

    for feat in model.feature_names_:
        if feat not in scores:
            scores[feat] = 1.0

    return scores

# Menghasilkan gambar graf struktur Bayesian Network / Naive Bayes yang menampilkan hubungan dependen kondisional P(a_i | v_j) dari variabel kelas V ke atribut-atribut fitur a_1, a_2, ..., a_n, serta menyimpannya melalui utils.save_figure
def plot_bayesian_network(model, path=None, top_n=12, target_name="is_canceled", figsize=(14, 7)):
    model._check_is_fitted()

    scores_dict = _compute_feature_discriminative_scores(model)
    ordered_features = sorted(
        model.feature_names_, key=lambda f: scores_dict.get(f, 0.0), reverse=True
    )
    if top_n is not None and top_n > 0:
        ordered_features = ordered_features[:top_n]

    n_feats = len(ordered_features)
    if n_feats == 0:
        raise ValueError("Tidak ada fitur untuk divisualisasikan pada graf Naive Bayes.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    root_x, root_y = 0.5, 0.84
    prior_str = " | ".join(
        [f"P(v={cls})={model.class_priors_[cls]:.2f}" for cls in model.classes_]
    )
    root_label = f"Target Class: {target_name}\n({prior_str})"

    root_box = FancyBboxPatch(
        (root_x - 0.19, root_y - 0.065),
        0.38,
        0.13,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=CHART_COLOR,
        edgecolor="#2b2b2b",
        linewidth=2.0,
        zorder=4,
    )
    ax.add_patch(root_box)
    ax.text(
        root_x,
        root_y,
        root_label,
        ha="center",
        va="center",
        color="white",
        fontsize=10.5,
        fontweight="bold",
        zorder=5,
    )

    n_row1 = (n_feats + 1) // 2
    n_row2 = n_feats - n_row1

    raw_vals = np.array([scores_dict.get(f, 1.0) for f in ordered_features], dtype=float)
    max_s = float(np.max(raw_vals)) if len(raw_vals) > 0 and np.max(raw_vals) > 0 else 1.0
    norm_scores = raw_vals / max_s

    cat_set = (
        set(model.categorical_model_.features_)
        if model.categorical_model_ is not None
        else set()
    )

    for idx, feat in enumerate(ordered_features):
        if idx < n_row1:
            row_idx = idx
            row_count = n_row1
            node_y = 0.36 if n_row2 > 0 else 0.25
        else:
            row_idx = idx - n_row1
            row_count = n_row2
            node_y = 0.12

        if row_count == 1:
            node_x = 0.5
        else:
            node_x = 0.08 + 0.84 * (row_idx / (row_count - 1))

        score_val = scores_dict.get(feat, 0.0)
        norm_w = norm_scores[idx]
        feat_type = "Cat" if feat in cat_set else "Num"

        arrow = FancyArrowPatch(
            (root_x, root_y - 0.065),
            (node_x, node_y + 0.055),
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>,head_length=7,head_width=4.5",
            color=CHART_COLOR,
            alpha=0.45 + 0.5 * norm_w,
            linewidth=1.2 + 2.2 * norm_w,
            zorder=2,
        )
        ax.add_patch(arrow)

        mid_x = 0.52 * root_x + 0.48 * node_x
        mid_y = 0.52 * (root_y - 0.065) + 0.48 * (node_y + 0.055)
        ax.text(
            mid_x,
            mid_y,
            f"P({feat[:10]}|V)",
            fontsize=7.0,
            color="#333333",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=3,
        )

        # Kotak node atribut a_i
        box_w = min(0.16, 0.82 / max(row_count, 1))
        box_h = 0.10
        fill_color = "#fce8ec" if feat_type == "Cat" else "#f2f4f8"
        border_color = CHART_COLOR if feat_type == "Cat" else "#3b5998"

        child_box = FancyBboxPatch(
            (node_x - box_w / 2, node_y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fill_color,
            edgecolor=border_color,
            linewidth=1.5,
            zorder=4,
        )
        ax.add_patch(child_box)

        disp_name = feat if len(feat) <= 16 else feat[:14] + ".."
        node_text = f"{disp_name}\n[{feat_type}] w={score_val:.2f}"
        ax.text(
            node_x,
            node_y,
            node_text,
            ha="center",
            va="center",
            color="#1f1f1f",
            fontsize=8.0,
            fontweight="bold",
            zorder=5,
        )

    ax.set_title(
        f"Struktur Graf Bayesian Network (Naive Bayes) — Top-{n_feats} Fitur\n"
        r"Model Hipotesis: $P(v_j \mid a_1, \dots, a_n) \propto P(v_j) \prod_i P(a_i \mid v_j)$",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    _save_figure(fig, path)
    return fig
