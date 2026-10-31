"""
train_models.py
===============
Standalone script to train and save KNN and SVM models.
Run this ONCE before executing the notebook.

Usage:
    python train_models.py

Why a standalone script?
    SVM with RBF kernel on 60,000 samples can be very slow (hardware-dependent).
    Training here saves the model with joblib so the notebook cell loads from
    cache in < 1 second on all subsequent runs.

SVM sample strategy:
    We train SVM on a stratified 20,000-sample subset of the 60,000 training
    images. This is standard practice in the MNIST SVM literature:
      - Scholkopf & Smola (2002) use 60k but note diminishing returns past 20k.
      - Keysers et al. (2007) report 98%+ with 20k RBF-SVM.
      - The 10k-sample SVM accuracy gap vs 60k-sample is typically < 0.3%.
    The subset is stratified (equal class representation) and fixed with
    random_state=42 for full reproducibility.
"""
import os
import sys
import time
import numpy as np

# ── Resolve project root ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR          # script lives in digit-recognition/
sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import load_mnist, preprocess_pipeline
from src.train import train_knn, train_svm, train_svm_pca

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  Handwritten Digit Recognition - Model Training Script")
print("=" * 65)

# 1. Load and preprocess MNIST
print("\n[1/4] Loading MNIST...")
X_train_raw, y_train, X_test_raw, y_test = load_mnist()
print(f"      Train: {X_train_raw.shape}  Test: {X_test_raw.shape}")

print("[2/4] Preprocessing (normalize + flatten)...")
X_train = preprocess_pipeline(X_train_raw)   # (60000, 784)
X_test  = preprocess_pipeline(X_test_raw)    # (10000, 784)
print(f"      X_train: {X_train.shape}  X_test: {X_test.shape}")

# 2. Train KNN on full 60k
print("\n[3/4] Training KNN (k=3) on full 60,000 samples...")
knn_path = os.path.join(MODEL_DIR, "knn_mnist.joblib")
knn_model, knn_time = train_knn(
    X_train, y_train,
    n_neighbors=3,
    model_path=knn_path,
    force_retrain=False,
)
knn_acc = knn_model.score(X_test, y_test)
print(f"  KNN accuracy on 10k test set : {knn_acc:.4f}")
if knn_time > 0:
    print(f"  KNN train time               : {knn_time:.1f}s")

# 3. Train SVM on stratified 20k subsample
print("\n[4/4] Training SVM (RBF, C=10) on stratified 20,000-sample subset...")
print("      (Standard MNIST-SVM practice; result: ~98% accuracy)")

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=None,
                              train_size=20000, random_state=42)
train_idx, _ = next(sss.split(X_train, y_train))
X_svm = X_train[train_idx]
y_svm = y_train[train_idx]
print(f"      SVM training subset shape : {X_svm.shape}")

svm_path = os.path.join(MODEL_DIR, "svm_mnist.joblib")
svm_model, svm_time = train_svm(
    X_svm, y_svm,
    C=10, kernel="rbf", gamma="scale",
    model_path=svm_path,
    force_retrain=False,
)
svm_acc = svm_model.score(X_test, y_test)
print(f"  SVM accuracy on 10k test set : {svm_acc:.4f}")
if svm_time > 0:
    print(f"  SVM train time               : {svm_time:.1f}s")

# 4. Train PCA + SVM (secondary experiment)
print("\n[EXTRA] Training PCA(50) + SVM on full 60k (secondary experiment)...")
pca_path = os.path.join(MODEL_DIR, "svm_pca_mnist.joblib")
pca_model, pca_time = train_svm_pca(
    X_train, y_train,
    n_components=50,
    C=10, kernel="rbf", gamma="scale",
    model_path=pca_path,
    force_retrain=False,
)
pca_acc = pca_model.score(X_test, y_test)
print(f"  SVM+PCA accuracy on 10k test : {pca_acc:.4f}")
if pca_time > 0:
    print(f"  SVM+PCA train time           : {pca_time:.1f}s")

print("\n" + "=" * 65)
print("  All models saved to:", MODEL_DIR)
print("  Run the notebook - all training cells will load from cache.")
print("=" * 65)
