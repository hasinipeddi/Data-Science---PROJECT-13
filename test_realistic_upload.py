import sys
import os
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"c:\DS PROJECT\digit-recognition")
import joblib
from src.preprocess import preprocess_canvas_image
from src.predict import predict_single

knn = joblib.load(r"c:\DS PROJECT\digit-recognition\models\knn_mnist.joblib")
svm = joblib.load(r"c:\DS PROJECT\digit-recognition\models\svm_mnist.joblib")

# Simulate a phone photo of a handwritten 5:
# Paper is light gray (225), digit is dark gray (30), canvas is 400x400, digit is in center
img = Image.new("RGB", (400, 400), color=(225, 225, 225))
d = ImageDraw.Draw(img)
# Draw digit 5 in center with standard 8px pen
d.line([(150, 100), (250, 100)], fill=(30, 30, 30), width=8)
d.line([(150, 100), (150, 180)], fill=(30, 30, 30), width=8)
d.line([(150, 180), (240, 180)], fill=(30, 30, 30), width=8)
d.line([(240, 180), (240, 260)], fill=(30, 30, 30), width=8)
d.line([(240, 260), (140, 260)], fill=(30, 30, 30), width=8)

arr = np.array(img)
vec = preprocess_canvas_image(arr)

k_pred, k_conf, _ = predict_single(knn, vec, "knn")
s_pred, s_conf, _ = predict_single(svm, vec, "svm")
dists, indices = knn.kneighbors(vec)
neighs = knn._y[indices[0]]

print("=== REALISTIC UPLOADED PHOTO OF DIGIT 5 (paper 225, ink 30) ===")
print("KNN Prediction:", k_pred, "Confidence:", k_conf, "Neighbours:", neighs, "Distances:", np.round(dists[0], 2))
print("SVM Prediction:", s_pred, "Confidence:", s_conf)
p28 = vec.reshape(28, 28)
print("Preprocessed min:", p28.min(), "max:", p28.max(), "mean:", p28.mean())
print("Border pixel intensity in preprocessed 28x28:", p28[0, :5])
