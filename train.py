import sys

if __name__ == "__main__" and "--baseline" not in sys.argv:
    from train_mixed import main as run_mixed_training
    run_mixed_training()
    sys.exit(0)


from utils import preprocess_image



from pathlib import Path

DATA_DIR = Path("data/BanglaLekha-Isolated/Images")

# Find all 84 character folders.
class_folders = sorted(
    [folder for folder in DATA_DIR.iterdir() if folder.is_dir()],
    key=lambda folder: int(folder.name)
)

# Count images in each folder.
total_images = 0

for folder in class_folders:
    image_count = len(list(folder.glob("*.png")))
    total_images += image_count

print("Number of classes:", len(class_folders))
print("Total images:", total_images)
print("First class:", class_folders[0].name)
print("Last class:", class_folders[-1].name)




import numpy as np






image_paths = []
labels = []

for folder in class_folders:
    class_id = int(folder.name)

    for image_path in folder.glob("*.png"):
        image_paths.append(str(image_path))

        # CNN labels must start from 0, not 1.
        labels.append(class_id - 1)

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Total image paths:", len(image_paths))
print("Total labels:", len(labels))
print("First label:", labels[0])
print("Last label:", labels[-1])



#Split our dataset-------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split

# First split: 80% training, 20% remaining.
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Second split: divide remaining 20% equally.
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)

print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Test images:", len(test_paths))




import tensorflow as tf

BATCH_SIZE = 32


def load_image(path, label):
    # Read and decode the PNG image as grayscale.
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_png(image_bytes, channels=1)

    # Use our shared preprocessing function.
    image = preprocess_image(image)

    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=5000, seed=42
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

for images, batch_labels in train_ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", batch_labels.shape)
    print("Pixel range:", float(tf.reduce_min(images)),
          float(tf.reduce_max(images)))



#architecture

from tensorflow.keras import layers, models

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        # Learn simple strokes and edges.
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Learn more complex character patterns.
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Learn higher-level character shapes.
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Convert features into a classification.
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),

        # Predict one of 84 character classes.
        layers.Dense(84, activation="softmax")
    ])

    return model

model = build_cnn()
model.summary()




model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("CNN compiled successfully!")





import mlflow

# Tell MLflow where to store experiment records.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create or select our experiment.
mlflow.set_experiment("Bangla_Character_Recognition")

print("MLflow configured successfully!")
print("Database: mlflow.db")




# Experiment 1: Original character images
EPOCHS = 3
STEPS_PER_EPOCH = 500
VALIDATION_STEPS = 100


if __name__ == "__main__":
    
    with mlflow.start_run(run_name="Run_1_Original_SharedPreprocessing"):

        # Record training settings
        mlflow.log_param("model", "Custom_CNN")
        mlflow.log_param("training_data", "original")
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("steps_per_epoch", STEPS_PER_EPOCH)


        # Train the CNN
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS
        )

        # Save the trained baseline model.
        model.save("baseline_cnn.keras")
        print("Baseline CNN saved successfully!")

        # Record results for every epoch
        for epoch in range(EPOCHS):
            mlflow.log_metric(
                "train_accuracy",
                history.history["accuracy"][epoch],
                step=epoch + 1
            )

            mlflow.log_metric(
                "train_loss",
                history.history["loss"][epoch],
                step=epoch + 1
            )
            mlflow.log_metric(
                "val_accuracy",
                history.history["val_accuracy"][epoch],
                step=epoch + 1
            )
            mlflow.log_metric(
                "val_loss",
                history.history["val_loss"][epoch],
                step=epoch + 1
            )

        print("Run 1 completed and logged to MLflow!")
