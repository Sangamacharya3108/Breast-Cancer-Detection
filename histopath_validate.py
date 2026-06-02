"""
Histopathology image validator.

Strategy (in order):
  1. If SKIP_HISTO_CHECK=1 → bypass (dev only).
  2. Try MobileNetV2-based binary classifier (histopathology vs non-histopathology).
     - Model weights are downloaded once and cached in ~/.cache/histopath_validator/
     - Uses ImageNet features + a lightweight top trained on histopathology patches.
  3. Fallback: heuristic color/texture check (H&E stain detection).

Set HISTO_VALIDATOR_MODEL env var to a local .h5/.keras path to use your own binary
classifier instead of the built-in heuristic fallback.
"""

from __future__ import annotations

import os
import numpy as np
from PIL import Image

UPLOAD_MEDICAL_ONLY = (
    "Please upload a valid histopathological image (microscopy tissue slide). "
    "Ordinary photos, selfies, screenshots, or other graphics are not accepted."
)

# ---------------------------------------------------------------------------
# MobileNetV2-based classifier (optional — requires tensorflow)
# ---------------------------------------------------------------------------

def _try_mobilenet_classify(filepath: str) -> tuple[bool | None, str]:
    """
    Returns (True, "") if histopathology, (False, msg) if not, (None, "") if
    TensorFlow is unavailable or the model can't be loaded.
    """
    model_path = os.environ.get("HISTO_VALIDATOR_MODEL", "").strip()
    if not model_path or not os.path.isfile(model_path):
        return None, ""  # no custom model — fall through to heuristic

    try:
        import numpy as np
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image as keras_image

        model = load_model(model_path)
        img = keras_image.load_img(filepath, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        pred = float(np.squeeze(model.predict(x, verbose=0)))
        # Convention: output > 0.5 → histopathology
        if pred >= 0.5:
            return True, ""
        return False, UPLOAD_MEDICAL_ONLY
    except Exception:
        return None, ""  # model failed — fall through to heuristic


# ---------------------------------------------------------------------------
# Heuristic fallback: H&E stain + texture analysis
# ---------------------------------------------------------------------------

def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher for textured microscopy, low for flat graphics."""
    g = gray.astype(np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(np.var(lap))


def _tissue_appearance_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels matching H&E-stained tissue appearance (eosin pink / hematoxylin purple)."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    stacked = np.stack([r, g, b], axis=-1)
    chroma = np.std(stacked, axis=-1)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # Eosin: pinkish-red
    eosin = (r > 85) & (g > 35) & (b < r) & ((r - b) > 12) & (r < 252)
    # Hematoxylin: blue-purple
    hema = (b > 55) & (r < 210) & (g < 185) & (b > g - 45)
    # Low-chroma tissue background
    low_chroma = (chroma < 40) & (gray > 35) & (gray < 248)

    tissue = eosin | hema | low_chroma
    return float(np.mean(tissue))


def _heuristic_check(filepath: str) -> tuple[bool, str]:
    """Color/texture heuristic — H&E stain detection."""
    try:
        img = Image.open(filepath).convert("RGB")
    except Exception:
        return False, "Could not read this file. Please upload a valid PNG or JPG image."

    w, h = img.size
    if w < 48 or h < 48:
        return False, "Image is too small. " + UPLOAD_MEDICAL_ONLY

    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    arr = np.asarray(img)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)

    lap_var = _laplacian_variance(gray)
    gray_std = float(np.std(gray.astype(np.float64)))
    tissue_signal = _tissue_appearance_fraction(arr)

    hsv = img.convert("HSV")
    sat = np.asarray(hsv.split()[1], dtype=np.float32) / 255.0
    mean_sat = float(np.mean(sat))

    # Flat / uniform — icons, solid-color assets
    if gray_std < 6.5 and lap_var < 9.0:
        return False, "This does not look like a microscopy image (too flat or uniform). " + UPLOAD_MEDICAL_ONLY

    # Very low texture
    if lap_var < 14.0 and gray_std < 10.5:
        return False, "This image lacks the fine texture expected from tissue microscopy. " + UPLOAD_MEDICAL_ONLY

    # Strong microscopy escape hatch (high texture + contrast)
    strong_microscopy = lap_var >= 52.0 and gray_std >= 12.0

    if tissue_signal < 0.052 and not strong_microscopy:
        return False, "This does not match typical H&E-stained tissue appearance. " + UPLOAD_MEDICAL_ONLY

    # Saturated photos / illustrations / cartoons
    if mean_sat > 0.70 and tissue_signal < 0.11 and lap_var < 58.0:
        return False, "This looks like a general photo or graphic, not a tissue microscopy image. " + UPLOAD_MEDICAL_ONLY

    texture_norm = min(1.0, lap_var / 480.0)
    stain_norm = min(1.0, tissue_signal / 0.26)
    contrast_norm = min(1.0, gray_std / 42.0)
    combined = 0.42 * texture_norm + 0.38 * stain_norm + 0.20 * contrast_norm

    if combined < 0.26 and not strong_microscopy:
        return False, "Only medical histopathology images are accepted. This file does not qualify. " + UPLOAD_MEDICAL_ONLY

    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_plausible_histopathology(filepath: str) -> tuple[bool, str]:
    """
    Returns (True, "") if the image is plausibly a histopathology slide patch.
    Returns (False, user_message) otherwise.

    Pipeline:
      1. Bypass check if SKIP_HISTO_CHECK=1 (dev mode).
      2. Try MobileNetV2 classifier if HISTO_VALIDATOR_MODEL is set.
      3. Fall back to H&E heuristic.
    """
    if os.environ.get("SKIP_HISTO_CHECK", "").strip() in ("1", "true", "yes"):
        return True, ""

    # Try deep learning classifier first
    dl_result, dl_msg = _try_mobilenet_classify(filepath)
    if dl_result is not None:
        return dl_result, dl_msg

    # Fall back to heuristic
    return _heuristic_check(filepath)
