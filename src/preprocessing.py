from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "dataset_weather"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

TEST_SIZE = 0.20

RANDOM_SEED = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_MAP = {
    "road_signs": 0,
    "pedestrians": 1
}


# ============================================================
# WEATHER CONDITIONS
# ============================================================

WEATHER_CONDITIONS = [
    "daylight",
    "rainy",
    "snow",
    "night"
]


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Load image, correct EXIF orientation,
    and convert to RGB.
    """

    with Image.open(image_path) as image:

        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        return image.copy()


# ============================================================
# RESIZE WITH ASPECT RATIO
# ============================================================

def resize_preserve_aspect(image):
    """
    Resize image to fit inside 128x128
    while preserving its original aspect ratio.
    """

    image = image.copy()

    image.thumbnail(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        IMAGE_SIZE,
        color=(0, 0, 0)
    )

    x = (IMAGE_SIZE[0] - image.width) // 2
    y = (IMAGE_SIZE[1] - image.height) // 2

    canvas.paste(
        image,
        (x, y)
    )

    return canvas


# ============================================================
# PREPROCESS ONE IMAGE
# ============================================================

def preprocess_image(image_path):
    """
    Complete preprocessing for one image:

    1. Load
    2. Preserve aspect ratio
    3. Resize and pad to 128x128
    4. Convert to grayscale
    5. Normalize pixels to 0-1
    6. Flatten into a feature vector
    """

    image = load_image(
        image_path
    )

    image = resize_preserve_aspect(
        image
    )

    # RGB -> grayscale
    image = image.convert("L")

    # Convert to NumPy
    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Normalize 0-255 -> 0-1
    image /= 255.0

    # 128 x 128 -> 16,384 features
    return image.reshape(-1)


# ============================================================
# FIND WEATHER IMAGES
# ============================================================

def collect_images():
    """
    Collect all images from:

    daylight
    rainy
    snow
    night

    and both classes.
    """

    samples = []

    for condition in WEATHER_CONDITIONS:

        for class_name, label in CLASS_MAP.items():

            folder = (
                DATASET_DIR
                / condition
                / class_name
            )

            if not folder.exists():
                raise FileNotFoundError(
                    f"Missing folder: {folder}"
                )

            images = sorted(
                [
                    path
                    for path in folder.iterdir()
                    if path.is_file()
                    and path.suffix.lower()
                    in VALID_EXTENSIONS
                ]
            )

            print(
                f"{condition:<10} "
                f"{class_name:<12}: "
                f"{len(images)} images"
            )

            for image_path in images:

                samples.append(
                    (
                        image_path,
                        label,
                        image_path.stem
                    )
                )

    return samples


# ============================================================
# GROUP SAMPLES BY ORIGINAL IMAGE
# ============================================================

def create_groups(samples):
    """
    Group the four weather versions of
    each original image together.

    Example:

    pedestrian_0001
        daylight
        rainy
        snow
        night

    All four remain in the same split.
    """

    groups = {}

    for image_path, label, image_id in samples:

        key = (
            label,
            image_id
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(
            image_path
        )

    return groups


# ============================================================
# GROUPED TRAIN / TEST SPLIT
# ============================================================

def split_groups(groups):
    """
    Split original images into training and testing
    groups while keeping all weather versions together.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    train_groups = []
    test_groups = []

    for label in sorted(CLASS_MAP.values()):

        class_groups = [
            key
            for key in groups
            if key[0] == label
        ]

        rng.shuffle(
            class_groups
        )

        test_count = max(
            1,
            int(
                len(class_groups)
                * TEST_SIZE
            )
        )

        test_groups.extend(
            class_groups[:test_count]
        )

        train_groups.extend(
            class_groups[test_count:]
        )

    return train_groups, test_groups


# ============================================================
# BUILD FEATURES FROM GROUPS
# ============================================================

def build_split(
    groups,
    selected_groups
):
    """
    Convert selected image groups into
    feature matrix X and label vector y.
    """

    X = []
    y = []

    for label, image_id in selected_groups:

        image_paths = groups[
            (label, image_id)
        ]

        for image_path in image_paths:

            features = preprocess_image(
                image_path
            )

            X.append(
                features
            )

            y.append(
                label
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    return X, y


# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

def save_split_info(
    train_groups,
    test_groups
):

    train_ids = [
        f"{label}:{image_id}"
        for label, image_id in train_groups
    ]

    test_ids = [
        f"{label}:{image_id}"
        for label, image_id in test_groups
    ]

    with open(
        OUTPUT_DIR / "split_info.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "GROUPED TRAIN / TEST SPLIT\n"
        )

        file.write(
            "==========================\n\n"
        )

        file.write(
            f"Train original images: "
            f"{len(train_groups)}\n"
        )

        file.write(
            f"Test original images: "
            f"{len(test_groups)}\n\n"
        )

        file.write(
            "TRAIN GROUPS\n"
        )

        file.write(
            "------------\n"
        )

        for item in sorted(train_ids):

            file.write(
                item + "\n"
            )

        file.write(
            "\nTEST GROUPS\n"
        )

        file.write(
            "-----------\n"
        )

        for item in sorted(test_ids):

            file.write(
                item + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("IMAGE PREPROCESSING")
    print("=" * 65)

    print(
        f"\nDataset: {DATASET_DIR}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}"
    )

    print(
        "Grayscale features: "
        f"{IMAGE_SIZE[0] * IMAGE_SIZE[1]}"
    )

    # --------------------------------------------------------
    # Collect images
    # --------------------------------------------------------

    print("\nCollecting images...")

    samples = collect_images()

    print(
        f"\nTotal images found: "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # Create groups
    # --------------------------------------------------------

    print(
        "\nCreating original-image groups..."
    )

    groups = create_groups(
        samples
    )

    print(
        f"Original images/groups: "
        f"{len(groups)}"
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    print(
        "\nCreating grouped train/test split..."
    )

    train_groups, test_groups = split_groups(
        groups
    )

    print(
        f"Training groups: "
        f"{len(train_groups)}"
    )

    print(
        f"Testing groups : "
        f"{len(test_groups)}"
    )

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    print(
        "\nPreprocessing training images..."
    )

    X_train, y_train = build_split(
        groups,
        train_groups
    )

    print(
        "Preprocessing testing images..."
    )

    X_test, y_test = build_split(
        groups,
        test_groups
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )

    # --------------------------------------------------------
    # Save combined dataset
    # --------------------------------------------------------

    X = np.concatenate(
        [X_train, X_test],
        axis=0
    )

    y = np.concatenate(
        [y_train, y_test],
        axis=0
    )

    np.save(
        OUTPUT_DIR / "X.npy",
        X
    )

    np.save(
        OUTPUT_DIR / "y.npy",
        y
    )

    # --------------------------------------------------------
    # Save split information
    # --------------------------------------------------------

    save_split_info(
        train_groups,
        test_groups
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETE")
    print("=" * 65)

    print("\nDataset shapes:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_test : {X_test.shape}"
    )

    print(
        f"y_test : {y_test.shape}"
    )

    print(
        f"\nTotal X: {X.shape}"
    )

    print(
        f"Total y: {y.shape}"
    )

    print("\nClass distribution:")

    print(
        "Road Signs  :",
        np.sum(y == 0)
    )

    print(
        "Pedestrians  :",
        np.sum(y == 1)
    )

    print("\nLabel mapping:")

    print(
        "0 = Road Sign"
    )

    print(
        "1 = Pedestrian"
    )

    print("\nSaved files:")

    print(
        "data/processed/X_train.npy"
    )

    print(
        "data/processed/y_train.npy"
    )

    print(
        "data/processed/X_test.npy"
    )

    print(
        "data/processed/y_test.npy"
    )

    print(
        "data/processed/split_info.txt"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()