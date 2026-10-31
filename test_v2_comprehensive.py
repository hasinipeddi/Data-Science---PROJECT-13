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

def preprocess_canvas_image_v2(image_array: np.ndarray) -> np.ndarray:
    if isinstance(image_array, np.ndarray):
        # Handle RGBA with transparency
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            # If alpha channel is present, composite on black or white background
            alpha = image_array[:, :, 3] / 255.0
            rgb = image_array[:, :, :3]
            # If the drawing is white strokes on transparent/black:
            if rgb.max() > 0 and (rgb * alpha[:, :, None]).mean() > 10:
                gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114) * alpha
                pil_img = Image.fromarray(gray.astype(np.uint8))
            else:
                pil_img = Image.fromarray(image_array.astype(np.uint8)).convert("RGB")
        else:
            pil_img = Image.fromarray(image_array.astype(np.uint8))
    else:
        pil_img = image_array

    gray = pil_img.convert("L")
    arr = np.array(gray, dtype=np.float64)

    # Blank image check
    if arr.max() == 0:
        return np.zeros((1, 784), dtype=np.float64)

    # 1. Background detection: sample outer border
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    border_val = np.median(border)

    # If border is light (>120), image is dark-on-light -> invert
    if border_val > 120:
        arr = 255.0 - arr
        # Background floor subtraction
        bg_floor = np.percentile(arr, 15)
        arr = np.maximum(0.0, arr - bg_floor)
    else:
        # Image is light-on-dark (e.g. canvas with black background)
        bg_floor = np.percentile(arr, 10)
        arr = np.maximum(0.0, arr - bg_floor)

    # 2. Dynamic threshold for bounding box
    max_val = arr.max()
    if max_val < 10:
        return np.zeros((1, 784), dtype=np.float64)

    # Threshold at 15% of peak intensity to ignore noise
    thresh = (arr > max(15.0, 0.15 * max_val)).astype(np.uint8)
    rows = np.any(thresh, axis=1)
    cols = np.any(thresh, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Normalize contrast to full 0..255 before resize
    if arr.max() > 0:
        arr = (arr / arr.max()) * 255.0

    # 3. Resize to 20x20 preserving aspect ratio
    h, w = arr.shape
    if h == 0 or w == 0:
        return np.zeros((1, 784), dtype=np.float64)

    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    pil_crop = Image.fromarray(arr.astype(np.uint8))
    pil_resized = pil_crop.resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(pil_resized, dtype=np.float64)

    # Clean residual low-level noise after interpolation
    if resized.max() > 0:
        resized = (resized / resized.max()) * 255.0
    resized[resized < 25.0] = 0.0

    # 4. Stroke thickness compensation:
    # In MNIST, digits have an average active pixel density of ~25-35% in the 20x20 box.
    # If the stroke is too thin (active ratio < 22%), dilate slightly to match MNIST felt-pen strokes.
    active_ratio = np.count_nonzero(resized > 40) / (20 * 20)
    if active_ratio < 0.22:
        resized = ndimage.grey_dilation(resized, size=(2, 2))
        if resized.max() > 0:
            resized = (resized / resized.max()) * 255.0

    # 5. Pad into 28x28
    canvas = np.zeros((28, 28), dtype=np.float64)
    y_off = (28 - resized.shape[0]) // 2
    x_off = (28 - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

    # 6. Center of mass centering to (14.0, 14.0)
    if canvas.sum() > 0:
        cy, cx = ndimage.center_of_mass(canvas)
        if not (np.isnan(cy) or np.isnan(cx)):
            shift_y = int(round(14.0 - cy))
            shift_x = int(round(14.0 - cx))
            # Limit shift to stay in bounds
            shift_y = max(-4, min(4, shift_y))
            shift_x = max(-4, min(4, shift_x))
            canvas = ndimage.shift(canvas, [shift_y, shift_x], cval=0.0)

    # 7. Normalize to [0.0, 1.0]
    if canvas.max() > 0:
        canvas = canvas / canvas.max()
    canvas = np.clip(canvas, 0.0, 1.0)

    return canvas.reshape(1, -1)

# Now test across all 10 digits with stroke widths 3, 5, 8, and paper colors 255, 230, 200!
print("=== COMPREHENSIVE TEST: ALL DIGITS 0-9 ACROSS CONDITIONS ===")
for stroke_w in [3, 6, 10]:
    for paper_color in [255, 220]:
        print(f"\n--- Stroke Width = {stroke_w}, Paper Gray Level = {paper_color} ---")
        k_correct = 0
        s_correct = 0
        for d in range(10):
            # Generate drawing on paper
            img = Image.new("RGB", (300, 300), color=(paper_color, paper_color, paper_color))
            d_raw = draw_digit(d, stroke_w=stroke_w)
            # Paste into img
            pil_d = Image.fromarray(d_raw).resize((200, 200))
            img.paste(pil_d, (50, 50))
            arr = np.array(img)

            vec = preprocess_canvas_image_v2(arr)
            k_pred, k_conf, _ = predict_single(knn, vec, "knn")
            s_pred, s_conf, _ = predict_single(svm, vec, "svm")
            dists, indices = knn.kneighbors(vec)
            neighs = knn._y[indices[0]]

            if k_pred == d: k_correct += 1
            if s_pred == d: s_correct += 1
            k_mark = "OK" if k_pred == d else "FAIL"
            s_mark = "OK" if s_pred == d else "FAIL"
            if d == 5 or k_mark == "FAIL":
                print(f"  Digit {d}: KNN={k_pred} ({k_mark}, neighs={neighs}, dist={dists[0][0]:.2f}) | SVM={s_pred} ({s_mark})")
        print(f"Total Correct: KNN = {k_correct}/10, SVM = {s_correct}/10")
