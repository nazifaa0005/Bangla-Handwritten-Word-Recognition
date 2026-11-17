
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import preprocess_image

DATA_DIR = Path("data/BanglaLekha-Isolated/Images")


def make_test_dataset():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR.resolve()}")

    image_paths, labels = [], []
    # Preserve the image enumeration logic of the original training script.
    folders = sorted(
        (p for p in DATA_DIR.iterdir() if p.is_dir()),
        key=lambda p: int(p.name),
    )
    for folder in folders:
        for path in folder.glob("*.png"):
            image_paths.append(str(path))
            labels.append(int(folder.name) - 1)

    if not image_paths:
        raise ValueError(f"No PNG images found under {DATA_DIR}")

    paths = np.array(image_paths)
    labels = np.array(labels)
    _, temp_paths, _, temp_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    _, test_paths, _, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42,
        stratify=temp_labels,
    )

    def load_image(path, label):
        image = tf.image.decode_png(tf.io.read_file(path), channels=1)
        return preprocess_image(image), label

    ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    return (ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(32).prefetch(tf.data.AUTOTUNE)), len(test_paths)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved CNN checkpoint")
    parser.add_argument("model_path", nargs="?", default="models/model.keras",
                        help="Path to saved .keras model (default: models/model.keras)")
    parser.add_argument("--model", dest="model_override", default=None,
                        help="Alternative named argument for the model path")
    parser.add_argument("--run-id", default=None,
                        help="Optional existing MLflow run ID to log the results into")
    args = parser.parse_args()

    model_path = Path(args.model_override or args.model_path)
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")

    test_ds, count = make_test_dataset()
    print(f"Evaluating {model_path} on {count} test images")
    model = tf.keras.models.load_model(str(model_path))
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    if args.run_id:
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        with mlflow.start_run(run_id=args.run_id):
            mlflow.log_metric("test_accuracy", float(test_accuracy))
            mlflow.log_metric("test_loss", float(test_loss))
        print(f"Results logged to MLflow run {args.run_id}")
    else:
        print("MLflow logging skipped (use --run-id to log to an existing run).")


if __name__ == "__main__":
    main()