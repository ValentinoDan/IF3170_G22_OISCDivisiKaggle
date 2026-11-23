from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def normalize_labels(y):
    y_arr = np.asarray(y).reshape(-1)
    unique = set(np.unique(y_arr).tolist())
    if unique == {0, 1}:
        return 2 * y_arr.astype(float) - 1.0
    if unique == {-1, 1}:
        return y_arr.astype(float)
    raise ValueError(f"y harus biner {{0, 1}} atau {{-1, 1}}, diterima: {sorted(unique)}")


def stratified_sample(X, y, limit, random_state=42):
    if limit is None or limit >= len(y):
        return X, y
    rng = np.random.default_rng(random_state)
    idx_neg = np.flatnonzero(y == -1)
    idx_pos = np.flatnonzero(y == 1)
    n_pos = max(1, min(len(idx_pos), round(limit * len(idx_pos) / len(y))))
    n_neg = max(1, min(len(idx_neg), limit - n_pos))
    chosen = np.concatenate([
        rng.choice(idx_neg, n_neg, replace=False),
        rng.choice(idx_pos, n_pos, replace=False),
    ])
    return X[chosen], y[chosen]


def plot_svm_2d(
    models,
    X,
    y,
    limit=None,
    sv_limit=None,
    grid_size=200,
    output_path=None,
    show=True,
    random_state=42,
    class_names=("Not Canceled (0)", "Canceled (1)"),
):
    if not isinstance(models, dict) or len(models) == 0:
        raise ValueError("models harus berupa dict tidak kosong berisi {nama: fitted_svm}.")

    X_arr = np.asarray(X, dtype=float)
    y_arr = normalize_labels(y)

    if X_arr.ndim != 2 or X_arr.shape[1] < 2:
        raise ValueError("X harus matriks 2D dengan minimal 2 fitur.")

    use_pca = X_arr.shape[1] > 2
    if use_pca:
        pca = PCA(n_components=2, random_state=random_state)
        X_2d = pca.fit_transform(X_arr)
        x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"
        y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)"
    else:
        pca = None
        X_2d = X_arr.copy()
        x_label, y_label = "Feature 1", "Feature 2"

    X_vis_2d, y_vis = stratified_sample(X_2d, y_arr, limit=limit, random_state=random_state)

    x_min, x_max = np.percentile(X_vis_2d[:, 0], [1, 99])
    y_min, y_max = np.percentile(X_vis_2d[:, 1], [1, 99])
    pad_x = max(0.5, 0.08 * (x_max - x_min))
    pad_y = max(0.5, 0.08 * (y_max - y_min))

    xx, yy = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, grid_size),
        np.linspace(y_min - pad_y, y_max + pad_y, grid_size),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_eval = pca.inverse_transform(grid_2d) if use_pca else grid_2d

    n_models = len(models)
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.2 * nrows), squeeze=False)
    axes = axes_grid.ravel()
    for unused_ax in axes[n_models:]:
        unused_ax.axis("off")

    pt_size = 6 if len(y_vis) > 2000 else 22
    pt_alpha = 0.25 if len(y_vis) > 2000 else 0.7
    rng = np.random.default_rng(random_state)

    for ax, (title, model) in zip(axes, models.items()):
        score_fn = getattr(model, "decision_function", None) or getattr(model, "_SVM__calculatePrediction")
        Z = np.asarray(score_fn(grid_eval), dtype=float).reshape(xx.shape)

        # 1. Area klasifikasi
        ax.contourf(
            xx, yy, Z >= 0,
            levels=[-0.5, 0.5, 1.5],
            colors=["#d6e6f4", "#f9d6d5"],
            alpha=0.45,
        )

        # 2. Bidang pemisah f(x)=0 (solid) dan margin f(x)=+-1 (dashed)
        z_min, z_max = float(np.min(Z)), float(np.max(Z))
        contour_levels = [lvl for lvl in (-1.0, 0.0, 1.0) if z_min < lvl < z_max]
        if contour_levels:
            styles = ["-" if lvl == 0.0 else "--" for lvl in contour_levels]
            widths = [2.0 if lvl == 0.0 else 1.2 for lvl in contour_levels]
            ax.contour(
                xx, yy, Z,
                levels=contour_levels,
                colors="black",
                linestyles=styles,
                linewidths=widths,
            )

        # 3. Sebaran titik data (proyeksi 2D)
        ax.scatter(
            X_vis_2d[y_vis == -1, 0], X_vis_2d[y_vis == -1, 1],
            c="#1f77b4", s=pt_size, alpha=pt_alpha, label=class_names[0],
        )
        ax.scatter(
            X_vis_2d[y_vis == 1, 0], X_vis_2d[y_vis == 1, 1],
            c="#d62728", s=pt_size, alpha=pt_alpha, label=class_names[1],
        )

        # 4. Support vectors asli dari fitted model (diproduksi/diproyeksikan ke 2D)
        sv = np.asarray(model.support_vectors, dtype=float)
        n_sv_total = len(sv)
        if n_sv_total > 0:
            sv_2d = pca.transform(sv) if use_pca else sv
            if sv_limit is not None and sv_limit < n_sv_total:
                sv_idx = rng.choice(n_sv_total, sv_limit, replace=False)
                sv_plot = sv_2d[sv_idx]
            else:
                sv_plot = sv_2d
            ax.scatter(
                sv_plot[:, 0], sv_plot[:, 1],
                s=pt_size * 3.5, facecolors="none", edgecolors="black",
                linewidths=0.8, alpha=0.5, label=f"Support Vectors ({n_sv_total:,})",
            )

        n_train_rows = len(model.alpha) if hasattr(model, "alpha") else len(y_arr)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_title(f"{title}\n(SV: {n_sv_total:,} / {n_train_rows:,})", fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Saved SVM 2D visualization to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
