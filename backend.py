"""Backend inference for emotion detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.preprocess import EMOTION_LABELS, preprocess_face


class EmotionDetector:
    """Loads a trained model and predicts emotions from images."""

    def __init__(self, model_path: str = "model/emotion_model.h5") -> None:
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train it first with `python train_model.py`."
            )

        self.model = load_model(model_path)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, bgr_image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return list(faces)

    def predict_emotion(self, bgr_image: np.ndarray) -> dict[str, Any]:
        """Predict emotion for the largest detected face in image."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            raise ValueError("No face detected in the provided image.")

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face = gray[y : y + h, x : x + w]
        resized = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
        model_input = preprocess_face(resized)

        probs = self.model.predict(model_input, verbose=0)[0]
        label_idx = int(np.argmax(probs))

        return {
            "emotion": EMOTION_LABELS[label_idx],
            "confidence": float(probs[label_idx]),
            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "probabilities": {EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)},
        }
