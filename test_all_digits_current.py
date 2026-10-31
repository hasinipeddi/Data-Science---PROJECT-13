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

def draw_digit(digit, stroke_w=4):
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    if digit == 0:
        d.ellipse([(50, 40), (150, 160)], outline=(0, 0, 0), width=stroke_w)
    elif digit == 1:
        d.line([(100, 40), (100, 160)], fill=(0, 0, 0), width=stroke_w)
    elif digit == 2:
        d.line([(50, 50), (140, 50)], fill=(0, 0, 0), width=stroke_w)
        d.line([(140, 50), (140, 100)], fill=(0, 0, 0), width=stroke_w)
        d.line([(140, 100), (50, 160)], fill=(0, 0, 0), width=stroke_w)
        d.line([(50, 160), (140, 160)], fill=(0, 0, 0), width=stroke_w)
    elif digit == 3:
        d.line([(50, 40), (140, 40)], fill=(0, 0, 0), width=stroke_w)
        d.line([(140, 40), (90, 100)], fill=(0, 0, 0), width=stroke_w)
        d.arc([(50, 100), (140, 160)], start=270, end=90, fill=(0, 0, 0), width=stroke_w)
    elif digit == 4:
        d.line([(120, 40), (60, 120)], fill=(0, 0, 0), width=stroke_w)
        d.line([(60, 120), (150, 120)], fill=(0, 0, 0), width=stroke_w)
        d.line([(120, 40), (120, 160)], fill=(0, 0, 0), width=stroke_w)
    elif digit == 5:
        d.line([(50, 40), (140, 40)], fill=(0, 0, 0), width=stroke_w)
        d.line([(50, 40), (50, 100)], fill=(0, 0, 0), width=stroke_w)
        d.line([(50, 100), (130, 100)], fill=(0, 0, 0), width=stroke_w)
        d.line([(130, 100), (130, 160)], fill=(0, 0, 0), width=stroke_w)
        d.line([(130, 160), (40, 160)], fill=(0, 0, 0), width=stroke_w)
    elif digit == 6:
        d.line([(130, 40), (60, 110)], fill=(0, 0, 0), width=stroke_w)
        d.ellipse([(50, 100), (140, 160)], outline=(0, 0, 0), width=stroke_w)
    elif digit == 7:
        d.line([(50, 40), (140, 40)], fill=(0, 0, 0), width=stroke_w)
        d.line([(140, 40), (70, 160)], fill=(0, 0, 0), width=stroke_w)
    elif digit == 8:
        d.ellipse([(60, 40), (140, 100)], outline=(0, 0, 0), width=stroke_w)
        d.ellipse([(55, 95), (145, 160)], outline=(0, 0, 0), width=stroke_w)
    elif digit == 9:
        d.ellipse([(60, 40), (140, 100)], outline=(0, 0, 0), width=stroke_w)
        d.line([(140, 70), (140, 160)], fill=(0, 0, 0), width=stroke_w)
    return np.array(img)

print("=== TESTING CURRENT PREPROCESSING (stroke_w = 4) ===")
for d in range(10):
    img_arr = draw_digit(d, stroke_w=4)
    vec = preprocess_canvas_image(img_arr)
    k_pred, k_conf, _ = predict_single(knn, vec, "knn")
    s_pred, s_conf, _ = predict_single(svm, vec, "svm")
    dists, indices = knn.kneighbors(vec)
    print(f"True {d} | KNN={k_pred} (conf={k_conf:.2f}, neighs={knn._y[indices[0]]}, dists={np.round(dists[0], 2)}) | SVM={s_pred} (conf={s_conf:.2f})")
