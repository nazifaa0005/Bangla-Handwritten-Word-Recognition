import numpy as np
import tensorflow as tf


def preprocess_image(image):
    # Convert pixels to float values.
    image = tf.cast(image, tf.float32)

    # Find the non-black pixels (handwriting).
    ink = tf.where(tf.reduce_any(image > 0, axis=-1))

    # Crop away empty space if handwriting exists.
    def crop_to_ink():
        top_left = tf.reduce_min(ink, axis=0)
        bottom_right = tf.reduce_max(ink, axis=0) + 1

        return image[
            top_left[0]:bottom_right[0],
            top_left[1]:bottom_right[1],
            :
        ]

    image = tf.cond(
        tf.shape(ink)[0] > 0,
        crop_to_ink,
        lambda: image
    )

    # Resize without stretching the character.
    image = tf.image.resize_with_pad(
        image, target_height=40, target_width=40
    )

    # Center on a 48x48 black canvas.
    image = tf.image.pad_to_bounding_box(
        image,
        offset_height=4,
        offset_width=4,
        target_height=48,
        target_width=48
    )

    # Normalize pixel values to 0–1.
    image = image / 255.0

    return image



def remove_matra(image):
    """Remove long upper horizontal strokes from a 48x48 image.

    Experimental augmentation, not guaranteed to remove only matra.
    Expects pixel values in the range 0 to 1.
    """
    import cv2
    import numpy as np

    array = image.numpy().squeeze()
    binary = (array > 110 / 255).astype(np.uint8) * 255

    upper = binary.copy()
    upper[29:, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    horizontal = cv2.morphologyEx(upper, cv2.MORPH_OPEN, kernel)
    
    
    
    # Find the first detected horizontal stroke from the top.
    rows = np.where(np.any(horizontal > 0, axis=1))[0]

    mask = np.zeros_like(horizontal, dtype=bool)

    if len(rows) > 0:
        first_row = rows[0]
        
        # Use the first detected horizontal line as a reference.
        # Inspect original pixels around that line, not just the OpenCV mask.
        selected = np.zeros_like(horizontal)

        band_start = max(0, first_row - 2)
        band_end = min(first_row + 5, horizontal.shape[0])

        selected[first_row:first_row + 6, :] = horizontal[
            first_row:first_row + 6, :
        ]

        mask = cv2.dilate(
            selected,
            np.ones((2, 1), dtype=np.uint8)
            ) > 0
    
    


    processed = array.copy()
    processed[mask] = 0

    return tf.convert_to_tensor(
        processed[..., np.newaxis],
        dtype=tf.float32
    )



if __name__ == "__main__":
    from pathlib import Path

    # Find one sample image from class 1.
    image_path = next(Path("data/BanglaLekha-Isolated/Images/39").glob("*.png"))
    

    # Read the original grayscale image.
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.image.decode_png(image_bytes, channels=1)

    # Apply our shared preprocessing function.
    processed = preprocess_image(image)

    print("Original shape:", image.shape)
    print("Processed shape:", processed.shape)
    print("Minimum pixel value:", float(tf.reduce_min(processed)))
    print("Maximum pixel value:", float(tf.reduce_max(processed)))
    
    
    
    
    from PIL import Image

    preview = (processed.numpy()[:, :, 0] * 255).astype("uint8")

    Image.fromarray(preview).resize((384, 384)).save(
        "preprocessing_preview.png"
    )

    print("Preview saved successfully!")
    
    
    
    from PIL import Image

    modified = remove_matra(processed)

    comparison = np.concatenate(
        [processed.numpy().squeeze(), modified.numpy().squeeze()],
        axis=1
    )

    Image.fromarray(
        (comparison * 255).astype("uint8")
    ).resize((960, 480)).save("matra_comparison.png")

    print("Matra comparison saved!")

    
    
    

