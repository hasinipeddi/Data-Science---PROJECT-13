import os
import sys
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
from src.preprocess import preprocess_canvas_image
from src.predict import predict_single

knn = joblib.load(os.path.join(PROJECT_ROOT, "models", "knn_mnist.joblib"))
svm = joblib.load(os.path.join(PROJECT_ROOT, "models", "svm_mnist.joblib"))

# Create synthetic images of digit 5
# Case 1: Black on white (standard scan / photo / drawing on paper)
img_b_on_w = Image.new("RGB", (200, 200), color=(255, 255, 255))
d = ImageDraw.Draw(img_b_on_w)
# draw a 5
d.line([(50, 40), (140, 40)], fill=(0, 0, 0), width=16)
d.line([(50, 40), (50, 100)], fill=(0, 0, 0), width=16)
d.line([(50, 100), (130, 100)], fill=(0, 0, 0), width=16)
d.line([(130, 100), (130, 160)], fill=(0, 0, 0), width=16)
d.line([(130, 160), (40, 160)], fill=(0, 0, 0), width=16)

arr_bw = np.array(img_b_on_w)
vec_bw = preprocess_canvas_image(arr_bw)

knn_d, knn_c, _ = predict_single(knn, vec_bw, "knn")
svm_d, svm_c, _ = predict_single(svm, vec_bw, "svm")
dists, indices = knn.kneighbors(vec_bw)
print("--- Synthetic 5 (Black on white) ---")
print(f"KNN={knn_d} (conf={knn_c:.2f}, neighs={knn._y[indices[0]]}, dists={np.round(dists[0], 2)})")
print(f"SVM={svm_d} (conf={svm_c:.2f})")

# Let's inspect the 28x28 preprocessed image
p28 = vec_bw.reshape(28, 28)
print("Preprocessed 28x28 stats: min=", p28.min(), "max=", p28.max(), "mean=", p28.mean())
print("Nonzero pixels count:", np.count_nonzero(p28 > 0.1))
