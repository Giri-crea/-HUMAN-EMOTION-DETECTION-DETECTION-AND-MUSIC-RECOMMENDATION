"""Train CNN emotion model on FER2013 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from utils.preprocess import load_fer2013


def build_emotion_cnn(input_shape: tuple[int, int, int] = (48, 48, 1), num_classes: int = 7) -> Sequential:
    """CNN architecture for FER2013 emotion recognition."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),
            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion detection model on FER2013")
    parser.add_argument(
        "--data-path",
        default="data/fer2013.csv",
        help="Path to FER2013 CSV file (default: data/fer2013.csv)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", default="model/emotion_model.h5", help="Model output path")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"FER2013 dataset not found at {data_path}. Download the fer2013.csv file and place it there."
        )

    x_train, x_val, y_train, y_val = load_fer2013(str(data_path))

    model = build_emotion_cnn()
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
