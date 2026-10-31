import sys
import os
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"c:\DS PROJECT\digit-recognition")
import joblib
from src.preprocess import load_mnist, preprocess_pipeline, preprocess_canvas_image
from src.predict import predict_single
from test_all_digits_current import draw_digit

knn = joblib.load(r"c:\DS PROJECT\digit-recognition\models\knn_mnist.joblib")
svm = joblib.load(r"c:\DS PROJECT\digit-recognition\models\svm_mnist.joblib")

print("=" * 70)
print("  VERIFICATION 1: ORIGINAL MNIST TEST SET ACCURACY (DIGITS 0-9)")
print("=" * 70)
_, _, X_test_raw, y_test = load_mnist()
X_test = preprocess_pipeline(X_test_raw)

# Evaluate on first 500 test samples
n_test = 500
knn_preds = knn.predict(X_test[:n_test])
svm_preds = svm.predict(X_test[:n_test])
knn_acc = np.mean(knn_preds == y_test[:n_test])
svm_acc = np.mean(svm_preds == y_test[:n_test])
print(f"First {n_test} MNIST test samples: KNN Accuracy = {knn_acc*100:.2f}%, SVM Accuracy = {svm_acc*100:.2f}%")

print("\n" + "=" * 70)
print("  VERIFICATION 2: SPECIFIC TEST ON DIGIT 5 (ACROSS STROKE WIDTHS & BACKGROUNDS)")
print("=" * 70)
for stroke_w in [2, 4, 6, 8, 14]:
    for bg_color in [255, 220, 190]:
        img = Image.new("RGB", (300, 300), color=(bg_color, bg_color, bg_color))
        d_raw = draw_digit(5, stroke_w=stroke_w)
        pil_d = Image.fromarray(d_raw).resize((180, 180))
        img.paste(pil_d, (60, 60))
        arr = np.array(img)

        vec = preprocess_canvas_image(arr)
        k_pred, k_conf, _ = predict_single(knn, vec, "knn")
        s_pred, s_conf, _ = predict_single(svm, vec, "svm")
        dists, indices = knn.kneighbors(vec)
        neighs = knn._y[indices[0]]

        status = "PASSED" if k_pred == 5 else "FAILED"
        print(f"Stroke {stroke_w:2d} | BG {bg_color} | KNN={k_pred} ({status}, conf={k_conf:.2f}, neighs={neighs}, dists={np.round(dists[0], 2)}) | SVM={s_pred} (conf={s_conf:.2f})")

print("\n" + "=" * 70)
print("  VERIFICATION 3: ALL DIGITS 0-9 TEST (UPLOADED/DRAWN)")
print("=" * 70)
for stroke_w in [4, 8]:
    print(f"\n--- Testing all digits 0-9 with stroke_width = {stroke_w} ---")
    correct_k = 0
    correct_s = 0
    for d in range(10):
        img_arr = draw_digit(d, stroke_w=stroke_w)
        vec = preprocess_canvas_image(img_arr)
        k_pred, k_conf, _ = predict_single(knn, vec, "knn")
        s_pred, s_conf, _ = predict_single(svm, vec, "svm")
        dists, indices = knn.kneighbors(vec)
        neighs = knn._y[indices[0]]

        if k_pred == d: correct_k += 1
        if s_pred == d: correct_s += 1
        k_res = "OK" if k_pred == d else "FAIL"
        s_res = "OK" if s_pred == d else "FAIL"
        print(f"True {d} -> KNN: {k_pred} ({k_res}, neighs={neighs}) | SVM: {s_pred} ({s_res})")
    print(f"Summary for stroke_width={stroke_w}: KNN={correct_k}/10 correct, SVM={correct_s}/10 correct")

print("\n" + "=" * 70)
print("  ALL VERIFICATIONS COMPLETE")
print("=" * 70)
