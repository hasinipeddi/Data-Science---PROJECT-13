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

# Test various stroke widths for digit 5
for stroke_w in [2, 4, 6, 8, 12, 16, 20]:
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.line([(50, 40), (140, 40)], fill=(0, 0, 0), width=stroke_w)
    d.line([(50, 40), (50, 100)], fill=(0, 0, 0), width=stroke_w)
    d.line([(50, 100), (130, 100)], fill=(0, 0, 0), width=stroke_w)
    d.line([(130, 100), (130, 160)], fill=(0, 0, 0), width=stroke_w)
    d.line([(130, 160), (40, 160)], fill=(0, 0, 0), width=stroke_w)

    vec = preprocess_canvas_image(np.array(img))
    knn_d, knn_c, _ = predict_single(knn, vec, "knn")
    svm_d, svm_c, _ = predict_single(svm, vec, "svm")
    dists, indices = knn.kneighbors(vec)
    neighs = knn._y[indices[0]]
    norm = np.linalg.norm(vec)
    print(f"Stroke width {stroke_w:2d} | Norm={norm:.2f} | KNN={knn_d} (neighs={neighs}, dists={np.round(dists[0], 2)}) | SVM={svm_d} (conf={svm_c:.2f})")
