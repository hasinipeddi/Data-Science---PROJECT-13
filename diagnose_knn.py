import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from src.preprocess import load_mnist, preprocess_pipeline, preprocess_canvas_image
from src.predict import predict_single

knn_path = os.path.join(PROJECT_ROOT, "models", "knn_mnist.joblib")
svm_path = os.path.join(PROJECT_ROOT, "models", "svm_mnist.joblib")

knn = joblib.load(knn_path)
svm = joblib.load(svm_path)

print("Loaded models successfully.")
print("KNN n_samples_fit:", knn.n_samples_fit_)
print("KNN metric:", knn.effective_metric_)

# Test on MNIST test 5s
X_train_raw, y_train, X_test_raw, y_test = load_mnist()
X_test = preprocess_pipeline(X_test_raw)

idx_5 = np.where(y_test == 5)[0][:10]
print("\n--- MNIST Test Set Digits (Actual 5s) ---")
for i, idx in enumerate(idx_5):
    vec = X_test[idx:idx+1]
    knn_d, knn_c, knn_scores = predict_single(knn, vec, "knn")
    svm_d, svm_c, svm_scores = predict_single(svm, vec, "svm")
    dists, indices = knn.kneighbors(vec)
    neigh_labels = knn._y[indices[0]]
    print(f"Sample {idx}: True=5 | KNN={knn_d} (conf={knn_c:.2f}, neighs={neigh_labels}, dists={np.round(dists[0], 2)}) | SVM={svm_d} (conf={svm_c:.2f})")
