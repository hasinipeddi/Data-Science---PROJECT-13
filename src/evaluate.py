"""
evaluate.py
===========
Evaluation utilities for the MNIST digit recognition project.

Functions
---------
evaluate_model(model, X_test, y_test)  – accuracy, precision, recall, F1,
                                          classification_report, prediction time
build_comparison_df(results)           – side-by-side DataFrame of both models
plot_comparison_bar(df, save_path)     – grouped bar chart saved to outputs/
plot_confusion_matrix(cm, title, ...)  – seaborn heatmap saved to outputs/
"""

import time
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving files
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Default output directory (resolved relative to this file's location)
_DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ---------------------------------------------------------------------------
# 1. Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """Compute full evaluation metrics for one model.

    Parameters
    ----------
    model      : fitted sklearn estimator with a .predict() method
    X_test     : ndarray  shape (N, 784) normalised + flattened
    y_test     : ndarray  shape (N,)     integer labels
    model_name : str      label for display

    Returns
    -------
    dict with keys:
        model_name, accuracy, precision_macro, recall_macro, f1_macro,
        precision_per_class, recall_per_class, f1_per_class,
        report (str), pred_time (float), y_pred (ndarray)
    """
    # --- Prediction time ---
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    # --- Scalar metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # --- Per-class metrics ---
    prec_pc = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_pc = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_pc = f1_score(y_test, y_pred, average=None, zero_division=0)

    # --- Full sklearn report ---
    report = classification_report(y_test, y_pred,
                                   target_names=[str(i) for i in range(10)])

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_per_class": prec_pc,
        "recall_per_class": rec_pc,
        "f1_per_class": f1_pc,
        "report": report,
        "pred_time": pred_time,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# 2. Side-by-side comparison DataFrame
# ---------------------------------------------------------------------------

def build_comparison_df(
    knn_results: dict,
    svm_results: dict,
    knn_train_time: float,
    svm_train_time: float,
) -> pd.DataFrame:
    """Build a tidy comparison DataFrame from two evaluate_model result dicts.

    Parameters
    ----------
    knn_results, svm_results : dicts returned by evaluate_model()
    knn_train_time, svm_train_time : floats (seconds)

    Returns
    -------
    pd.DataFrame  with columns: Model, Accuracy, Precision(Macro),
                  Recall(Macro), F1(Macro), Train Time(s), Pred Time(s)
    """
    rows = []
    for res, t_train in [(knn_results, knn_train_time),
                          (svm_results, svm_train_time)]:
        rows.append({
            "Model": res["model_name"],
            "Accuracy": round(res["accuracy"], 4),
            "Precision (Macro)": round(res["precision_macro"], 4),
            "Recall (Macro)": round(res["recall_macro"], 4),
            "F1 (Macro)": round(res["f1_macro"], 4),
            "Train Time (s)": round(t_train, 2),
            "Pred Time (s)": round(res["pred_time"], 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Grouped bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    df: pd.DataFrame,
    save_path: str = None,
    dpi: int = 300,
) -> None:
    """Save a grouped bar chart comparing KNN and SVM metrics.

    Parameters
    ----------
    df        : DataFrame from build_comparison_df()
    save_path : str  full path to save PNG; defaults to outputs/comparison_bar.png
    dpi       : int
    """
    if save_path is None:
        save_path = os.path.join(_DEFAULT_OUT_DIR, "comparison_bar.png")

    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C72B0", "#DD8452"]

    for idx, row in df.iterrows():
        values = [row[m] for m in metrics]
        ax.bar(x + idx * width, values, width, label=row["Model"],
               color=colors[idx % 2], alpha=0.85)
        for xi, v in zip(x + idx * width, values):
            ax.text(xi, v + 0.003, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("KNN vs SVM — Performance Comparison on MNIST Test Set",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved comparison bar chart → {save_path}")


# ---------------------------------------------------------------------------
# 4. Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = None,
    dpi: int = 300,
) -> np.ndarray:
    """Plot and save a seaborn confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : ndarrays of integer labels
    title          : str
    save_path      : str  defaults to outputs/confusion_<title>.png
    dpi            : int

    Returns
    -------
    cm : ndarray  shape (10, 10) raw confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if save_path is None:
        safe_title = title.lower().replace(" ", "_")
        save_path = os.path.join(_DEFAULT_OUT_DIR, f"confusion_{safe_title}.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        linewidths=0.5, linecolor="lightgray", ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved confusion matrix → {save_path}")
    return cm


# ---------------------------------------------------------------------------
# 5. Misclassification grid
# ---------------------------------------------------------------------------

def plot_misclassified_grid(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_samples: int = 30,
    save_path: str = None,
    dpi: int = 300,
) -> None:
    """Plot a grid of misclassified test images with true vs predicted labels.

    Parameters
    ----------
    X_test     : ndarray  shape (N, 784) normalised features
    y_true     : ndarray  true labels
    y_pred     : ndarray  predicted labels
    model_name : str      used in title and filename
    n_samples  : int      number of errors to display
    save_path  : str
    dpi        : int
    """
    if save_path is None:
        safe_name = model_name.lower().replace(" ", "_").replace("+", "plus")
        save_path = os.path.join(_DEFAULT_OUT_DIR, f"errors_{safe_name}.png")

    # Indices of errors
    error_idx = np.where(y_true != y_pred)[0]
    n_show = min(n_samples, len(error_idx))
    np.random.seed(42)
    chosen = np.random.choice(error_idx, n_show, replace=False)

    cols = 10
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, idx in enumerate(chosen):
        img = X_test[idx].reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"T:{y_true[idx]}\nP:{y_pred[idx]}",
                          fontsize=7, color="red", pad=2)
        axes[i].axis("off")

    fig.suptitle(f"{model_name} — Misclassified Images (True → Predicted)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved misclassified grid → {save_path}")


# ---------------------------------------------------------------------------
# 6. Robustness bar chart
# ---------------------------------------------------------------------------

def plot_robustness(
    robustness_df: pd.DataFrame,
    save_path: str = None,
    dpi: int = 300,
) -> None:
    """Plot accuracy-drop bar chart for robustness test.

    Parameters
    ----------
    robustness_df : DataFrame with columns: Condition, KNN Accuracy, SVM Accuracy
    save_path     : str
    dpi           : int
    """
    if save_path is None:
        save_path = os.path.join(_DEFAULT_OUT_DIR, "robustness_comparison.png")

    conditions = robustness_df["Condition"].tolist()
    knn_accs = robustness_df["KNN Accuracy"].tolist()
    svm_accs = robustness_df["SVM Accuracy"].tolist()

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_knn = ax.bar(x - width / 2, knn_accs, width,
                      label="KNN (k=3)", color="#4C72B0", alpha=0.85)
    bars_svm = ax.bar(x + width / 2, svm_accs, width,
                      label="SVM (RBF)", color="#DD8452", alpha=0.85)

    for bar in bars_knn:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_svm:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Test Condition", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Robustness Test — KNN vs SVM under Augmented Conditions",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved robustness chart → {save_path}")
