"""
preprocess.py
=============
Data loading and preprocessing utilities for the MNIST digit recognition project.

Functions
---------
load_mnist()        - Load MNIST via keras.datasets; returns (X_train, y_train, X_test, y_test)
normalize(X)        - Scale pixel values from [0, 255] to [0.0, 1.0]
flatten(X)          - Reshape (N, 28, 28) to (N, 784)
preprocess_pipeline - Combined normalize + flatten in one call
preprocess_canvas_image - MNIST-style preprocessing for a user-drawn / uploaded image
"""

import numpy as np
from scipy import ndimage
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_mnist():
    """Load the MNIST dataset using keras.datasets (or local mnist.npz).

    Returns
    -------
    X_train : ndarray, shape (60000, 28, 28)  uint8 pixel values [0..255]
    y_train : ndarray, shape (60000,)          integer labels [0..9]
    X_test  : ndarray, shape (10000, 28, 28)  uint8 pixel values [0..255]
    y_test  : ndarray, shape (10000,)          integer labels [0..9]
    """
    import os
    local_npz = os.path.join(os.path.dirname(__file__), "..", "mnist.npz")
    if os.path.exists(local_npz):
        with np.load(local_npz, allow_pickle=True) as f:
            X_train, y_train = f["x_train"], f["y_train"]
            X_test, y_test = f["x_test"], f["y_test"]
            return X_train, y_train, X_test, y_test

    from tensorflow.keras.datasets import mnist  # type: ignore
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# 2. Normalisation
# ---------------------------------------------------------------------------

def normalize(X: np.ndarray) -> np.ndarray:
    """Normalize pixel values to the range [0.0, 1.0].

    Divides by 255.0 (the maximum uint8 value).  This reduces the influence
    of absolute stroke thickness and lighting variation between writers.

    Parameters
    ----------
    X : ndarray  – any shape, dtype uint8 or float

    Returns
    -------
    ndarray  – same shape, dtype float64
    """
    return X.astype(np.float64) / 255.0


# ---------------------------------------------------------------------------
# 3. Flattening
# ---------------------------------------------------------------------------

def flatten(X: np.ndarray) -> np.ndarray:
    """Flatten images from (N, 28, 28) → (N, 784).

    Scikit-learn estimators require 2-D input (samples × features).
    Flattening each 28×28 image to a 784-dimensional vector is the standard
    first step for KNN and SVM on MNIST.

    Parameters
    ----------
    X : ndarray, shape (N, 28, 28)

    Returns
    -------
    ndarray, shape (N, 784)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


# ---------------------------------------------------------------------------
# 4. Combined pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(X: np.ndarray) -> np.ndarray:
    """Apply normalize then flatten in one call.

    Parameters
    ----------
    X : ndarray, shape (N, 28, 28)  uint8

    Returns
    -------
    ndarray, shape (N, 784)  float64, values in [0.0, 1.0]
    """
    return flatten(normalize(X))


# ---------------------------------------------------------------------------
# 5. MNIST-style preprocessing for the Streamlit app
# ---------------------------------------------------------------------------

def _center_by_mass(img: np.ndarray) -> np.ndarray:
    """Shift a 28×28 image so that its centre of mass aligns with the frame centre (14.0, 14.0).

    This replicates the MNIST preprocessing step described in LeCun et al. (1998).
    """
    if img.sum() == 0:
        return img
    cy, cx = ndimage.center_of_mass(img)
    if np.isnan(cy) or np.isnan(cx):
        return img
    shift_y = int(round(14.0 - cy))
    shift_x = int(round(14.0 - cx))
    shift_y = max(-4, min(4, shift_y))
    shift_x = max(-4, min(4, shift_x))
    return ndimage.shift(img, [shift_y, shift_x], cval=0.0)


def preprocess_canvas_image(image_array: np.ndarray) -> np.ndarray:
    """Convert a raw RGBA / RGB / greyscale image array to a 28×28 MNIST-style array.

    Robust pipeline:
    1. Handles RGBA/RGB/Greyscale numpy arrays or PIL Images.
    2. Robust background detection via image perimeter sampling (handles white paper,
       gray scans, colored paper, or black drawing canvas).
    3. Noise floor suppression to ensure clean black (0.0) background as in MNIST.
    4. Adaptive foreground thresholding for bounding box calculation.
    5. Aspect-ratio preserving resize into a 20×20 box using Lanczos resampling.
    6. Stroke thickness calibration: if the stroke is thin (< 20% active pixel density),
       applies gentle grey dilation so stroke width matches MNIST's felt-pen strokes (~3px).
    7. Centers the 20×20 digit into a 28×28 canvas by centre of mass (14.0, 14.0).
    8. Normalizes intensity to [0.0, 1.0], matching the MNIST training distribution.
    9. Returns shape (1, 784) ready for predict().

    Parameters
    ----------
    image_array : ndarray – raw image (H, W, C) or (H, W)

    Returns
    -------
    ndarray, shape (1, 784)  float64 in [0.0, 1.0]
    """
    if isinstance(image_array, np.ndarray):
        # Handle RGBA with transparency
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            alpha = image_array[:, :, 3] / 255.0
            rgb = image_array[:, :, :3]
            # If drawing is white strokes on dark/transparent:
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

    # Empty image check
    if arr.max() == 0:
        return np.zeros((1, 784), dtype=np.float64)

    # 1. Background detection: sample outer border pixels
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    border_val = np.median(border)

    # If border is light (> 120), image is dark-on-light (paper) -> invert
    if border_val > 120:
        arr = 255.0 - arr
        # Background floor subtraction
        bg_floor = np.percentile(arr, 15)
        arr = np.maximum(0.0, arr - bg_floor)
    else:
        # Image is light-on-dark (drawing canvas)
        bg_floor = np.percentile(arr, 10)
        arr = np.maximum(0.0, arr - bg_floor)

    # 2. Dynamic threshold for bounding box
    max_val = arr.max()
    if max_val < 10:
        return np.zeros((1, 784), dtype=np.float64)

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

    # 3. Resize to 20×20 preserving aspect ratio
    h, w = arr.shape
    if h == 0 or w == 0:
        return np.zeros((1, 784), dtype=np.float64)

    scale = 20.0 / max(h, w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    pil_crop = Image.fromarray(arr.astype(np.uint8))
    pil_resized = pil_crop.resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(pil_resized, dtype=np.float64)

    # Clean residual low-level interpolation noise
    if resized.max() > 0:
        resized = (resized / resized.max()) * 255.0
    resized[resized < 25.0] = 0.0

    # 4. Stroke thickness compensation:
    # In MNIST, digits have an average active pixel density of ~25-35% in the 20x20 box.
    # If the stroke is thin (< 20% active density), apply gentle grey dilation to match MNIST strokes.
    active_ratio = np.count_nonzero(resized > 40) / (20 * 20)
    if active_ratio < 0.20:
        resized = ndimage.grey_dilation(resized, size=(3, 3))
        if resized.max() > 0:
            resized = (resized / resized.max()) * 255.0

    # 5. Pad into 28×28 canvas
    canvas = np.zeros((28, 28), dtype=np.float64)
    y_off = (28 - resized.shape[0]) // 2
    x_off = (28 - resized.shape[1]) // 2
    canvas[y_off:y_off + resized.shape[0], x_off:x_off + resized.shape[1]] = resized

    # 6. Centre by centre of mass (14.0, 14.0)
    canvas = _center_by_mass(canvas)

    # 7. Normalize to [0.0, 1.0]
    if canvas.max() > 0:
        canvas = canvas / canvas.max()
    canvas = np.clip(canvas, 0.0, 1.0)

    # 8. Flatten to (1, 784)
    return canvas.reshape(1, -1)


# ---------------------------------------------------------------------------
# Augmentation helpers (used in robustness section of the notebook)
# ---------------------------------------------------------------------------

def add_gaussian_noise(X: np.ndarray, sigma: float = 0.1,
                        random_state: int = 42) -> np.ndarray:
    """Add Gaussian noise to normalised images.

    Parameters
    ----------
    X          : ndarray  shape (N, 784), values in [0, 1]
    sigma      : float    standard deviation of the noise
    random_state : int

    Returns
    -------
    ndarray – same shape, clipped to [0, 1]
    """
    rng = np.random.RandomState(random_state)
    noisy = X + rng.normal(0, sigma, X.shape)
    return np.clip(noisy, 0.0, 1.0)


def add_rotation(X: np.ndarray, angle: float = 15.0,
                 random_state: int = 42) -> np.ndarray:
    """Randomly rotate each image by up to ±angle degrees.

    Parameters
    ----------
    X      : ndarray  shape (N, 784)
    angle  : float    max rotation in degrees
    random_state : int

    Returns
    -------
    ndarray – same shape
    """
    rng = np.random.RandomState(random_state)
    angles = rng.uniform(-angle, angle, X.shape[0])
    result = np.zeros_like(X)
    for i, (img_flat, a) in enumerate(zip(X, angles)):
        img_2d = img_flat.reshape(28, 28)
        rotated = ndimage.rotate(img_2d, a, reshape=False, cval=0.0)
        result[i] = rotated.flatten()
    return result


def add_shift(X: np.ndarray, max_shift: int = 3,
              random_state: int = 42) -> np.ndarray:
    """Randomly shift each image by up to max_shift pixels in both axes.

    Parameters
    ----------
    X          : ndarray  shape (N, 784)
    max_shift  : int      max pixel shift per axis
    random_state : int

    Returns
    -------
    ndarray – same shape
    """
    rng = np.random.RandomState(random_state)
    shifts = rng.randint(-max_shift, max_shift + 1, (X.shape[0], 2))
    result = np.zeros_like(X)
    for i, (img_flat, s) in enumerate(zip(X, shifts)):
        img_2d = img_flat.reshape(28, 28)
        shifted = ndimage.shift(img_2d, s, cval=0.0)
        result[i] = shifted.flatten()
    return result
