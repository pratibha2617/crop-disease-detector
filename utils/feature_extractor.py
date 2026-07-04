import cv2
import numpy as np

def extract_features(img_bgr, size=(64, 64)):
    """
    Extract color histogram + texture (LBP-like) features from a leaf image.
    Returns a 1D numpy array of features.
    """
    # Resize to fixed size
    img = cv2.resize(img_bgr, size)

    # ── Feature 1: HSV Color Histogram ──────────────────────────
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([img_hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([img_hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([img_hsv], [2], None, [32], [0, 256]).flatten()

    # Normalize histograms
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    hist_v = hist_v / (hist_v.sum() + 1e-7)

    # ── Feature 2: RGB Mean & Std per channel ────────────────────
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_stats = []
    for c in range(3):
        channel = img_rgb[:, :, c].astype(np.float32) / 255.0
        rgb_stats.extend([channel.mean(), channel.std()])

    # ── Feature 3: Grayscale texture (gradient magnitude) ────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    texture_stats = [
        magnitude.mean() / 255.0,
        magnitude.std() / 255.0,
        np.percentile(magnitude, 75) / 255.0
    ]

    # ── Combine all features ─────────────────────────────────────
    features = np.concatenate([
        hist_h, hist_s, hist_v,
        rgb_stats,
        texture_stats
    ])
    return features.astype(np.float32)
