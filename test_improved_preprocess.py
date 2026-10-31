import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import scipy.ndimage as ndimage

sys.path.insert(0, r"c:\DS PROJECT\digit-recognition")
import joblib
from src.predict import predict_single
from test_all_digits_current import draw_digit

knn = joblib.load(r"c:\DS PROJECT\digit-recognition\models\knn_mnist.joblib")
svm = joblib.load(r"c:\DS PROJECT\digit-recognition\models\svm_mnist.joblib")

def preprocess_improved(image_array: np.ndarray) -> np.ndarray:
    # 1. To PIL Image & grayscale
    if isinstance(image_array, np.ndarray):
        pil_img = Image.fromarray(image_array.astype(np.uint8))
    else:
        pil_img = image_array
    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float64)

    # 2. Invert if background is bright
    # Sample borders to determine true background color
    border_pixels = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    if border_pixels.mean() > 127:
        arr = 255.0 - arr

    # Subtract background floor so background is pure black (0)
    bg_floor = np.percentile(arr, 5)
    arr = np.maximum(0.0, arr - bg_floor)

    # Threshold noise
    if arr.max() > 0:
        arr[arr < (0.15 * arr.max())] = 0.0

    # 3. Crop to bounding box
    thresh = (arr > 20).astype(np.uint8)
    rows = np.any(thresh, axis=1)
    cols = np.any(thresh, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Normalize intensity before resizing so stroke has full contrast
    if arr.max() > 0:
        arr = (arr / arr.max()) * 255.0

    # 4. Resize to fit in 20x20
    h, w = arr.shape
    if h == 0 or w == 0:
        return np.zeros((1, 784), dtype=np.float64)

    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    pil_crop = Image.fromarray(arr.astype(np.uint8))
    pil_resized = pil_crop.resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(pil_resized, dtype=np.float64)

    # Re-normalize contrast after resize
    if resized.max() > 0:
        resized = (resized / resized.max()) * 255.0

    # If the stroke is thin (low mass), apply slight grey dilation to match MNIST felt-pen thickness
    active_ratio = np.count_nonzero(resized > 50) / (20 * 20)
    if active_ratio < 0.22:
        # Thicken stroke slightly
        resized = ndimage.grey_dilation(resized, size=(2, 2))

    # 5. Pad to 28x28
    canvas = np.zeros((28, 28), dtype=np.float64)
    y_off = (28 - resized.shape[0]) // 2
    x_off = (28 - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

    # 6. Center of mass centering
    if canvas.sum() > 0:
        cy, cx = ndimage.center_of_mass(canvas)
        if not (np.isnan(cy) or np.isnan(cx)):
            shift_y = int(round(14.0 - cy))
            shift_x = int(round(14.0 - cx))
            canvas = ndimage.shift(canvas, [shift_y, shift_x], cval=0.0)

    # 7. Normalize to [0, 1]
    if canvas.max() > 0:
        canvas = canvas / canvas.max()
    canvas = np.clip(canvas, 0.0, 1.0)

    return canvas.reshape(1, -1)

print("=== TESTING IMPROVED PREPROCESSING ACROSS STROKE WIDTHS 2, 4, 8 ===")
for stroke_w in [2, 4, 8]:
    print(f"\n--- STROKE WIDTH = {stroke_w} ---")
    correct_k, correct_s = 0, 0
    for d in range(10):
        img_arr = draw_digit(d, stroke_w=stroke_w)
        vec = preprocess_improved(img_arr)
        k_pred, k_conf, _ = predict_single(knn, vec, "knn")
        s_pred, s_conf, _ = predict_single(svm, vec, "svm")
        dists, indices = knn.kneighbors(vec)
        neighs = knn._y[indices[0]]
        k_ok = "OK" if k_pred == d else "FAIL"
        s_ok = "OK" if s_pred == d else "FAIL"
        if k_pred == d: correct_k += 1
        if s_pred == d: correct_s += 1
        print(f"True {d} | KNN={k_pred} ({k_ok}, conf={k_conf:.2f}, neighs={neighs}, dist={dists[0][0]:.2f}) | SVM={s_pred} ({s_ok}, conf={s_conf:.2f})")
    print(f"Summary for stroke_w={stroke_w}: KNN={correct_k}/10, SVM={correct_s}/10")
