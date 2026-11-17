
from pathlib import Path
import json

import cv2
import numpy as np
import tensorflow as tf

from segment_word import segment_word
from utils import preprocess_image

MODEL_PATH = Path("models/model.keras")
IMAGE_FOLDER = Path("segmentation_tests")
MAPPING_PATH = Path("character_map.json")


def load_character_mapping():
    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        int(item["id"]): item["character"]
        for item in data["bengali_character_sheet"]
    }


def predict_word(image_path, model, mapping):
    # Read a handwritten word (black ink on white background).
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Invert so handwriting becomes white on black.
    image = 255 - image

    # Split the word into character regions.
    crops, info = segment_word(image, min_h_frac=0.40)

    if not crops:
        return "", []

    predictions = []

    for crop in crops:
        # Our preprocessing expects height x width x 1.
        character = np.expand_dims(crop, axis=-1)

        processed = preprocess_image(character)
        processed = np.expand_dims(processed.numpy(), axis=0)

        # CNN returns probabilities for 84 classes.
        probabilities = model.predict(processed, verbose=0)[0]

        class_index = int(np.argmax(probabilities))

        # Training labels are 0-83; dataset IDs are 1-84.
        class_id = class_index + 1

        predictions.append({
            "character": mapping.get(class_id, "?"),
            "class_id": class_id,
            "confidence": float(probabilities[class_index])
        })

    word = "".join(item["character"] for item in predictions)

    return word, predictions


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    mapping = load_character_mapping()

    print("\nOur CNN — Handwritten Word Recognition")
    print("-" * 55)

    images = sorted(IMAGE_FOLDER.glob("*.png"))

    for image_path in images:
        try:
            word, predictions = predict_word(image_path, model, mapping)

            print(
                f"{image_path.name:12} "
                f"{len(predictions)} units -> {word}"
            )

        except Exception as error:
            print(f"{image_path.name}: ERROR - {error}")

    print("-" * 55)


if __name__ == "__main__":
    main()
