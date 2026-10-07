from pathlib import Path

import joblib
import numpy as np

from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "test_images"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

# Class mapping used during training
CLASS_NAMES = {
    0: "Road Sign",
    1: "Pedestrian"
}


# Supported image formats
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("MANUAL IMAGE PREDICTION")
print("=" * 70)

pca_path = MODEL_DIR / "pca.joblib"
mlp_path = MODEL_DIR / "mlp_sgd.joblib"

if not pca_path.exists():
    raise FileNotFoundError(
        f"PCA model not found:\n{pca_path}"
    )

if not mlp_path.exists():
    raise FileNotFoundError(
        f"MLP model not found:\n{mlp_path}"
    )


print("\nLoading PCA model...")
pca = joblib.load(pca_path)

print("Loading MLP-SGD model...")
mlp = joblib.load(mlp_path)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Apply the same preprocessing used during training:

    Image
      ↓
    RGB
      ↓
    Preserve aspect ratio
      ↓
    128 × 128 padding
      ↓
    Grayscale
      ↓
    Normalize 0–1
      ↓
    Flatten
    """

    image = Image.open(image_path)

    # Correct image orientation from EXIF metadata
    image = ImageOps.exif_transpose(image)

    # Convert to RGB
    image = image.convert("RGB")

    # Create black 128 × 128 canvas
    canvas = Image.new(
        "RGB",
        IMAGE_SIZE,
        (0, 0, 0)
    )

    # Preserve aspect ratio
    image.thumbnail(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    # Center the image
    x_offset = (
        IMAGE_SIZE[0] - image.width
    ) // 2

    y_offset = (
        IMAGE_SIZE[1] - image.height
    ) // 2

    canvas.paste(
        image,
        (x_offset, y_offset)
    )

    # Convert to grayscale
    gray = ImageOps.grayscale(canvas)

    # Convert to NumPy
    array = np.asarray(
        gray,
        dtype=np.float32
    )

    # Normalize to 0–1
    array /= 255.0

    # Flatten 128 × 128 → 16,384
    features = array.reshape(1, -1)

    return features


# ============================================================
# FIND TEST IMAGES
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"\nTest images folder not found:\n{TEST_DIR}\n\n"
        "Create a folder named 'test_images' "
        "in the project root and add images."
    )


image_paths = sorted(
    [
        path
        for path in TEST_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
)


if len(image_paths) == 0:
    raise ValueError(
        "\nNo images found inside test_images."
    )


print(
    f"\nFound {len(image_paths)} test images."
)


# ============================================================
# PREDICT EACH IMAGE
# ============================================================

results = []

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)


for image_path in image_paths:

    try:

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image_features = preprocess_image(
            image_path
        )

        # ----------------------------------------------------
        # PCA transformation
        # ----------------------------------------------------

        image_pca = pca.transform(
            image_features
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            mlp.predict(image_pca)[0]
        )

        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        probabilities = mlp.predict_proba(
            image_pca
        )[0]

        confidence = float(
            probabilities[prediction]
        )

        # ----------------------------------------------------
        # Class probabilities
        # ----------------------------------------------------

        road_sign_probability = float(
            probabilities[0]
        )

        pedestrian_probability = float(
            probabilities[1]
        )

        predicted_class = CLASS_NAMES[
            prediction
        ]

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "image": image_path.name,
                "class": prediction,
                "prediction": predicted_class,
                "confidence": confidence
            }
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\n" + "-" * 70)

        print(
            f"Image       : {image_path.name}"
        )

        print(
            f"Class       : {prediction}"
        )

        print(
            f"Prediction  : {predicted_class}"
        )

        print(
            f"Confidence  : "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Road Sign   : "
            f"{road_sign_probability * 100:.2f}%"
        )

        print(
            f"Pedestrian  : "
            f"{pedestrian_probability * 100:.2f}%"
        )

    except Exception as error:

        print("\n" + "-" * 70)

        print(
            f"Image       : {image_path.name}"
        )

        print(
            f"ERROR       : {error}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)


road_sign_count = sum(
    result["class"] == 0
    for result in results
)

pedestrian_count = sum(
    result["class"] == 1
    for result in results
)


print(
    f"\nImages processed : {len(results)}"
)

print(
    f"Road Signs       : {road_sign_count}"
)

print(
    f"Pedestrians      : {pedestrian_count}"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(
    f"\nOriginal features : "
    f"{pca.n_features_in_}"
)

print(
    f"PCA components    : "
    f"{pca.n_components_}"
)

print(
    f"MLP architecture  : "
    f"{mlp.hidden_layer_sizes}"
)

print(
    f"Learning rate     : "
    f"{mlp.learning_rate_init}"
)

print(
    f"Momentum          : "
    f"{mlp.momentum}"
)

print("\n" + "=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)