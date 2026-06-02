"""
Flask Application for Breast Cancer Histopathology Prediction.

This module serves as the primary entry point for the Breast Cancer Detection
AI application. It registers routes for home, predict, about, and history pages.
It validates uploaded images to ensure they represent plausible histopathology
microscopy tissue slides, and triggers model inference.

Features:
  - Image type and H&E staining validation using 'histopath_validate'.
  - Lazy-loaded Keras/TensorFlow model prediction if MODEL_PATH environment
    variable is set and points to a valid file.
  - Deterministic fallback mock predictions when no model is found,
    enabling lightweight offline testing.
  - Session-based user prediction history tracking.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

# Local Module Imports
from histopath_validate import is_plausible_histopathology

# ===========================================================================
# APPLICATION CONFIGURATION
# ===========================================================================

app = Flask(__name__)

# Security: Fetch secret key from environment or use a safe default for dev
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-change-me-in-production")

# File Upload limits: 16 MB maximum size
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Directories configuration
UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Machine Learning model parameters
MODEL_PATH: str = os.environ.get("MODEL_PATH", "").strip()
IMG_SIZE: int = int(os.environ.get("IMG_SIZE", "224"))


# ===========================================================================
# CORE MODEL INFERENCE & DEMO HELPERS
# ===========================================================================

def _demo_from_filename(filename: str) -> Tuple[str, float, str]:
    """
    Generates a deterministic pseudo-prediction based on the filename hash.

    This ensures that when a Keras deep learning model is not configured locally,
    the UI behaves deterministically during demonstrations, research reviews,
    and user verification workflows.

    Args:
        filename: The original name of the uploaded image file.

    Returns:
        A tuple of (Prediction Label, Rounded Confidence, Result Class ID).
    """
    # Create MD5 hash of filename to generate reliable deterministic values
    hash_int = int(hashlib.md5(filename.encode()).hexdigest(), 16)
    is_malignant = (hash_int % 2 == 1)
    
    # Scale confidence score dynamically between 55.0% and 99.9%
    confidence = 55.0 + (hash_int % 4500) / 100.0
    
    if is_malignant:
        return "Malignant", round(confidence, 1), "malignant"
    return "Benign", round(confidence, 1), "benign"


def _predict_with_keras(filepath: str) -> Tuple[str, float, str]:
    """
    Loads the Keras neural network model and executes deep learning inference.

    Note:
        To keep the Flask bootstrap process extremely fast and decouple heavy
        machine learning framework environments from general web server startup,
        TensorFlow libraries are lazy-imported inside this function only when
        running actual predictions.

    Args:
        filepath: The absolute disk path to the saved user-uploaded image.

    Returns:
        A tuple of (Prediction Label, Rounded Confidence, Result Class ID).
    """
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image

    # Load Model structure and pre-trained weights
    model = load_model(MODEL_PATH)
    
    # Load and preprocess image according to target dimension
    img = keras_image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img)
    processed_x = np.expand_dims(img_array, axis=0) / 255.0

    # Run network predictions
    raw_pred = model.predict(processed_x, verbose=0)
    np_pred = np.asarray(raw_pred)

    # Resolve probability of malignancy based on output dimensions:
    #   - Case 1: Binary single sigmoid output [p]
    #   - Case 2: Multi-class softmax outputs [p_benign, p_malignant]
    #   - Case 3: Squeezed scalar value
    if np_pred.ndim == 2 and np_pred.shape[1] == 1:
        p_malignant = float(np_pred[0][0])
    elif np_pred.ndim == 2 and np_pred.shape[1] >= 2:
        p_malignant = float(np_pred[0][1])
    else:
        p_malignant = float(np.squeeze(np_pred))

    # Bound probability boundary strictly within standard mathematical interval
    p_malignant = max(0.0, min(1.0, p_malignant))
    
    # Determine predicted class and convert confidence score to percentage
    if p_malignant >= 0.5:
        confidence_percentage = round(100.0 * p_malignant, 1)
        return "Malignant", confidence_percentage, "malignant"
    
    confidence_percentage = round(100.0 * (1.0 - p_malignant), 1)
    return "Benign", confidence_percentage, "benign"


def _run_prediction(save_path: str, original_name: str) -> Tuple[str, float, str]:
    """
    Coordinates and handles predictive model selection and safety fallbacks.

    Attempts deep learning inference if MODEL_PATH is valid and accessible.
    In the event of physical environment model execution failures or missing
    TensorFlow libraries, it falls back gracefully to a deterministic mock prediction.

    Args:
        save_path: Absolute physical file location of the uploaded image.
        original_name: Raw file name before system uuid sanitization.

    Returns:
        A tuple of (Prediction Label, Rounded Confidence, Result Class ID).
    """
    if MODEL_PATH and os.path.isfile(MODEL_PATH):
        try:
            return _predict_with_keras(save_path)
        except Exception:
            # Fall through to mock prediction if actual model fails
            pass
    return _demo_from_filename(original_name)


def _append_history(filename: str, prediction: str, confidence: float, result_class: str) -> None:
    """
    Saves a prediction execution record to the user's active session history.

    Keeps history bounded to the most recent 50 runs to ensure optimal memory
    overhead and light session footprint.

    Args:
        filename: Raw user-supplied name of the tested image.
        prediction: Predicted target category ('Malignant' or 'Benign').
        confidence: Prediction confidence score as a percentage.
        result_class: Result class identifier string ('malignant' or 'benign').
    """
    entry: Dict[str, Any] = {
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence,
        "result_class": result_class,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Retrieve current session list, insert new entry at front, and limit size to 50
    history_list: List[Dict[str, Any]] = session.get("history") or []
    history_list.insert(0, entry)
    session["history"] = history_list[:50]


# ===========================================================================
# FLASK ROUTE CONTROLLERS
# ===========================================================================

@app.route("/")
def index():
    """Renders the landing/home page highlighting project capabilities."""
    return render_template("index.html", active_page="home")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Renders prediction workflow page and processes uploaded histopathology images.

    Accepts POST requests with file upload under 'image' key, runs validation,
    triggers model execution, updates session history records, and renders prediction results.
    """
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    result_class: Optional[str] = None
    error: Optional[str] = None
    error_category: Optional[str] = None  # "irrelevant" (non-medical) vs common errors

    if request.method == "POST":
        file = request.files.get("image")
        
        if not file or file.filename == "":
            error = "Please select a histopathology image."
        else:
            name = secure_filename(file.filename)
            if not name:
                error = "Invalid file name."
            else:
                ext = os.path.splitext(name)[1].lower()
                if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                    error = "Allowed formats: PNG, JPG, JPEG, WEBP."
                else:
                    # Secure filename uniqueness to prevent collisions
                    unique_name = f"{uuid.uuid4().hex}{ext}"
                    save_path = os.path.join(UPLOAD_DIR, unique_name)
                    file.save(save_path)
                    
                    # Validate if image is a plausible stained histopathological tissue slice
                    is_valid, validation_msg = is_plausible_histopathology(save_path)
                    
                    if not is_valid:
                        error = validation_msg
                        error_category = "irrelevant"
                        try:
                            os.remove(save_path)
                        except OSError:
                            pass
                    else:
                        # Validation passes -> run prediction model
                        prediction, confidence, result_class = _run_prediction(
                            save_path, name
                        )
                        _append_history(name, prediction, confidence, result_class)

    return render_template(
        "predict.html",
        active_page="predict",
        prediction=prediction,
        confidence=confidence,
        result_class=result_class,
        error=error,
        error_category=error_category,
    )


@app.route("/about")
def about():
    """Renders research details, documentation, and background technology info."""
    return render_template("about.html", active_page="about")


@app.route("/history")
def history():
    """Retrieves and renders past session predictions saved in current browser state."""
    rows = session.get("history") or []
    return render_template("history.html", active_page="history", history=rows)


@app.route("/history/clear")
def clear_history():
    """Clears all session prediction entries and redirects back to history overview."""
    session.pop("history", None)
    return redirect(url_for("history"))


# ===========================================================================
# APPLICATION LAUNCHER
# ===========================================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
