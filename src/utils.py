import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CHART = "#062e78"

def parent(path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_figure(fig, path):
    if path is not None:
        parent(path)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_table(df, path):
    parent(path)
    df.to_csv(path)

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

def plot_target_distribution(y, path=None):
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color=CHART)
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value}\n({value / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_title("Distribusi Target")
    ax.set_xlabel("is_canceled")
    ax.set_ylabel("Jumlah")
    ax.set_ylim(0, counts.max() * 1.18)
    save_figure(fig, path)
    return counts

def plot_missing_values(df, path=None, top_n=20):
    missing = df.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(missing) + 1)))
    if len(missing) == 0:
        ax.text(0.5, 0.5, "Tidak ada missing values", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(missing.index[::-1], missing.values[::-1], color=CHART)
        ax.set_xlabel("Persentase missing (%)")
        ax.set_title("Missing Values per Kolom")
    save_figure(fig, path)
    return missing

def plot_correlation_heatmap(df, path=None, figsize=(12, 9)):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Korelasi Antar Fitur Numerik")
    save_figure(fig, path)
    return corr

def plot_feature_importance(scores, path=None, top_n=20):
    scores = pd.Series(scores).sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(scores) + 1)))
    ax.barh(scores.index, scores.values, color=CHART)
    ax.set_xlabel("Mutual Information")
    ax.set_title("Skor Mutual Information Fitur")
    save_figure(fig, path)
    return scores

def plot_explained_variance(pca, path=None):
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(ratios) + 1), ratios, color=CHART, alpha=0.85, label="Explained variance")
    ax.plot(range(1, len(ratios) + 1), cumulative, color="black", marker="o", label="Kumulatif")
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Proporsi Variance")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    save_figure(fig, path)
    return pd.DataFrame({"explained_variance": ratios, "cumulative": cumulative})