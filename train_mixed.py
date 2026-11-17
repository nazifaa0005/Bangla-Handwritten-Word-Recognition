from pathlib import Path
import numpy as np
import tensorflow as tf
import mlflow

from train import build_cnn, train_paths, train_labels, val_ds, BATCH_SIZE
from utils import preprocess_image, remove_matra

EPOCHS = 50
STEPS_PER_EPOCH = 500
VALIDATION_STEPS = 100
OUTPUT_DIR = Path("experimental_runs")


def apply_matra_removal(image):
    """Apply the existing OpenCV-based matra removal augmentation."""
    modified = remove_matra(tf.convert_to_tensor(image))
    return modified.numpy().astype(np.float32)


def load_mixed_image(path, label):
    image = tf.image.decode_png(tf.io.read_file(path), channels=1)
    image = preprocess_image(image)

    def modify():
        result = tf.py_function(
            func=apply_matra_removal,
            inp=[image],
            Tout=tf.float32,
        )
        result.set_shape((48, 48, 1))
        return result

    image = tf.cond(tf.random.uniform(()) < 0.5, modify, lambda: image)
    return image, label


class MLflowMetrics(tf.keras.callbacks.Callback):
    """Record metrics as each epoch finishes, including when training stops early."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        names = {
            "accuracy": "train_accuracy",
            "loss": "train_loss",
            "val_accuracy": "val_accuracy",
            "val_loss": "val_loss",
        }
        metrics = {
            output_name: float(logs[input_name])
            for input_name, output_name in names.items()
            if input_name in logs
        }
        if metrics:
            mlflow.log_metrics(metrics, step=epoch + 1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mixed_train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    mixed_train_ds = mixed_train_ds.shuffle(buffer_size=5000, seed=42)
    mixed_train_ds = mixed_train_ds.map(
        load_mixed_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    mixed_train_ds = mixed_train_ds.batch(BATCH_SIZE)
    mixed_train_ds = mixed_train_ds.prefetch(tf.data.AUTOTUNE)

    mixed_model = build_cnn()
    mixed_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = OUTPUT_DIR / "mixed_cnn_best_new_run.keras"
    final_model = OUTPUT_DIR / "mixed_cnn_final_new_run.keras"

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Bangla_Character_Recognition")

    with mlflow.start_run(run_name="Mixed_CNN_Reproduction"):
        mlflow.log_params({
            "model": "Custom_CNN",
            "training_data": "50%_original_50%_matra_modified",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": 0.001,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "validation_steps": VALIDATION_STEPS,
            "early_stopping_patience": 8,
        })

        history = mixed_model.fit(
            mixed_train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint),
                    monitor="val_accuracy",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    mode="max",
                    patience=8,
                    restore_best_weights=True,
                    verbose=1,
                ),
                MLflowMetrics(),
            ],
        )

        # Save only if training returned successfully; neither path is the shipped model.
        mixed_model.save(str(final_model))
        mlflow.log_metric("epochs_completed", len(history.history.get("loss", [])))
        if history.history.get("val_accuracy"):
            mlflow.log_metric("best_val_accuracy", max(history.history["val_accuracy"]))

    print("Mixed CNN experiment finished.")
    print("Best checkpoint:", checkpoint)
    print("Final restored model:", final_model)
    print("The submitted models/model.keras was not modified.")


if __name__ == "__main__":
    main()
