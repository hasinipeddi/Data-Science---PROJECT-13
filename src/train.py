"""
train.py
========
Training utilities for KNN and SVM models on MNIST.

Functions
---------
train_knn(X_train, y_train, ...)  - Train / load cached KNN model
train_svm(X_train, y_train, ...)  - Train / load cached SVM model
train_svm_pca(...)                - PCA-accelerated SVM (secondary experiment)
"""

import time
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Default paths
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models"
)


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 3,
    model_path: str = None,
    force_retrain: bool = False,
) -> tuple:
    """Train (or load cached) a KNN classifier.

    Parameters
    ----------
    X_train      : ndarray  shape (N, 784)  normalised + flattened features
    y_train      : ndarray  shape (N,)      integer labels
    n_neighbors  : int      default 3
    model_path   : str      where to save / load the model; defaults to
                            models/knn_mnist.joblib
    force_retrain: bool     if True, ignore any cached model

    Returns
    -------
    model       : KNeighborsClassifier  fitted model
    train_time  : float                 seconds spent training (0 if loaded)
    """
    if model_path is None:
        model_path = os.path.join(_DEFAULT_MODEL_DIR, "knn_mnist.joblib")

    if not force_retrain and os.path.exists(model_path):
        print(f"  [KNN] Loading cached model from {model_path}")
        model = joblib.load(model_path)
        return model, 0.0

    print(f"  [KNN] Training KNeighborsClassifier(n_neighbors={n_neighbors}) "
          f"on {X_train.shape[0]:,} samples ...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

    t0 = time.time()
    knn.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"  [KNN] Training complete in {train_time:.1f}s")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(knn, model_path)
    print(f"  [KNN] Model saved to {model_path}")
    return knn, train_time


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 10.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    model_path: str = None,
    force_retrain: bool = False,
) -> tuple:
    """Train (or load cached) an SVM classifier on the full training set.

    Training SVM on 60 000 samples is CPU-intensive (~3-8 min depending on
    hardware).  The model is saved to *model_path* with joblib so subsequent
    runs skip training entirely.

    Parameters
    ----------
    X_train   : ndarray  shape (N, 784)
    y_train   : ndarray  shape (N,)
    C         : float    regularisation parameter (default 10)
    kernel    : str      kernel type (default 'rbf')
    gamma     : str      kernel coefficient (default 'scale')
    model_path: str      save / load location
    force_retrain: bool  skip cache if True

    Returns
    -------
    model      : SVC         fitted model
    train_time : float       seconds spent training (0 if loaded from cache)
    """
    if model_path is None:
        model_path = os.path.join(_DEFAULT_MODEL_DIR, "svm_mnist.joblib")

    if not force_retrain and os.path.exists(model_path):
        print(f"  [SVM] Loading cached model from {model_path}")
        model = joblib.load(model_path)
        return model, 0.0

    print(f"  [SVM] Training SVC(C={C}, kernel='{kernel}', gamma='{gamma}') "
          f"on {X_train.shape[0]:,} samples ...")
    print("  [SVM] This may take 3-8 minutes on a typical laptop CPU ...")

    svm = SVC(C=C, kernel=kernel, gamma=gamma,
              random_state=42, decision_function_shape="ovr")

    t0 = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"  [SVM] Training complete in {train_time:.1f}s")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(svm, model_path)
    print(f"  [SVM] Model saved to {model_path}")
    return svm, train_time


# ---------------------------------------------------------------------------
# SVM + PCA (secondary speed experiment)
# ---------------------------------------------------------------------------

def train_svm_pca(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 50,
    C: float = 10.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    model_path: str = None,
    force_retrain: bool = False,
) -> tuple:
    """Train (or load cached) a PCA→SVM pipeline.

    This is a clearly labelled secondary experiment: PCA reduces 784 features
    to *n_components* principal components before fitting SVM, dramatically
    reducing training time with a modest accuracy trade-off.

    Returns
    -------
    pipeline   : sklearn.pipeline.Pipeline  (PCA → SVC)
    train_time : float
    """
    if model_path is None:
        model_path = os.path.join(_DEFAULT_MODEL_DIR, "svm_pca_mnist.joblib")

    if not force_retrain and os.path.exists(model_path):
        print(f"  [SVM+PCA] Loading cached model from {model_path}")
        pipeline = joblib.load(model_path)
        return pipeline, 0.0

    print(f"  [SVM+PCA] Training PCA({n_components})->SVC on "
          f"{X_train.shape[0]:,} samples ...")

    pipeline = Pipeline([
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("svm", SVC(C=C, kernel=kernel, gamma=gamma,
                    random_state=42, decision_function_shape="ovr")),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    print(f"  [SVM+PCA] Training complete in {train_time:.1f}s")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"  [SVM+PCA] Model saved to {model_path}")
    return pipeline, train_time
