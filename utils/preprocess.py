"""Preprocessing utilities for FER2013 emotion detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def _pixels_to_array(pixel_string: str) -> np.ndarray:
    """Convert FER2013 pixel string to a (48, 48, 1) float array in [0, 1]."""
    pixels = np.fromstring(pixel_string, sep=" ", dtype=np.float32)
    if pixels.size != 48 * 48:
        raise ValueError("Unexpected FER2013 image size; expected 48x48 pixels")
    image = pixels.reshape((48, 48, 1))
    return image / 255.0


def load_fer2013(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load FER2013 CSV and return train/validation arrays."""
    df = pd.read_csv(csv_path)
    required_cols = {"emotion", "pixels"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"FER2013 CSV must include columns: {required_cols}")

    x = np.stack(df["pixels"].map(_pixels_to_array).to_numpy())
    y = to_categorical(df["emotion"].astype(int).to_numpy(), num_classes=7)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=df["emotion"].astype(int).to_numpy(),
    )
    return x_train, x_val, y_train, y_val


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    """Prepare a single grayscale face crop for model prediction."""
    if gray_face.shape != (48, 48):
        raise ValueError("Input face must be a 48x48 grayscale image")
    face = gray_face.astype(np.float32) / 255.0
    return np.expand_dims(face, axis=(0, -1))
