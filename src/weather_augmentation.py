from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "dataset_crct"
OUTPUT_DIR = BASE_DIR / "data" / "dataset_weather"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (256, 256)

RANDOM_SEED = 42

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

CLASSES = {
    "pedestrians": "pedestrian",
    "road_signs": "road_sign"
}

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
    """Load image and convert it to RGB."""

    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


# ============================================================
# RESIZE IMAGE
# ============================================================

def resize_image(image):
    """Resize image while preserving aspect ratio."""

    image.thumbnail(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        IMAGE_SIZE,
        (0, 0, 0)
    )

    x = (IMAGE_SIZE[0] - image.width) // 2
    y = (IMAGE_SIZE[1] - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


# ============================================================
# DAYLIGHT
# ============================================================

def create_daylight(image):
    """Keep the original image as the daylight condition."""

    return image.copy()


# ============================================================
# RAIN
# ============================================================

def create_rain(image):
    """
    Simulate rainy conditions using:
    - darker exposure
    - slight blue/gray tint
    - rain streaks
    - light atmospheric blur
    """

    image = ImageEnhance.Brightness(image).enhance(0.75)

    image = ImageEnhance.Contrast(image).enhance(0.90)

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # Slight cool/blue tint
    array[:, :, 0] *= 0.85
    array[:, :, 1] *= 0.92
    array[:, :, 2] *= 1.05

    array = np.clip(
        array,
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(
        array,
        mode="RGB"
    )

    # Add rain streaks
    draw = ImageDraw.Draw(
        image,
        "RGBA"
    )

    width, height = image.size

    for _ in range(80):

        x = random.randint(
            0,
            width - 1
        )

        y = random.randint(
            0,
            height - 1
        )

        length = random.randint(
            8,
            20
        )

        draw.line(
            (x, y, x - 3, y + length),
            fill=(210, 225, 235, 100),
            width=1
        )

    image = image.filter(
        ImageFilter.GaussianBlur(radius=0.5)
    )

    return image


# ============================================================
# SNOW / WINTER
# ============================================================

def create_snow(image):
    """
    Simulate winter/snow conditions using:
    - increased brightness
    - reduced contrast
    - cool tint
    - snow particles
    """

    image = ImageEnhance.Brightness(image).enhance(1.15)

    image = ImageEnhance.Contrast(image).enhance(0.85)

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # Cool winter tint
    array[:, :, 0] *= 0.92
    array[:, :, 1] *= 0.98
    array[:, :, 2] *= 1.08

    array = np.clip(
        array,
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(
        array,
        mode="RGB"
    )

    # Add snow particles
    draw = ImageDraw.Draw(
        image,
        "RGBA"
    )

    width, height = image.size

    for _ in range(120):

        x = random.randint(
            0,
            width - 1
        )

        y = random.randint(
            0,
            height - 1
        )

        radius = random.randint(
            1,
            3
        )

        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ),
            fill=(255, 255, 255, 180)
        )

    return image


# ============================================================
# NIGHT
# ============================================================

def create_night(image):
    """
    Simulate night conditions using:
    - reduced brightness
    - reduced saturation
    - increased contrast
    - blue night tint
    """

    image = ImageEnhance.Brightness(image).enhance(0.35)

    image = ImageEnhance.Contrast(image).enhance(1.15)

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # Blue night tint
    array[:, :, 0] *= 0.55
    array[:, :, 1] *= 0.70
    array[:, :, 2] *= 1.10

    array = np.clip(
        array,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        array,
        mode="RGB"
    )


# ============================================================
# APPLY WEATHER CONDITION
# ============================================================

def apply_condition(image, condition):

    if condition == "daylight":
        return create_daylight(image)

    if condition == "rainy":
        return create_rain(image)

    if condition == "snow":
        return create_snow(image)

    if condition == "night":
        return create_night(image)

    raise ValueError(
        f"Unknown condition: {condition}"
    )


# ============================================================
# PROCESS ONE CLASS
# ============================================================

def process_class(
    class_name,
    prefix
):

    source_folder = SOURCE_DIR / class_name

    if not source_folder.exists():
        raise FileNotFoundError(
            f"Missing source folder: {source_folder}"
        )

    images = sorted(
        [
            path
            for path in source_folder.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in VALID_EXTENSIONS
        ]
    )

    print(f"\n{class_name}")
    print("-" * 40)
    print(f"Source images: {len(images)}")

    for condition in WEATHER_CONDITIONS:

        output_folder = (
            OUTPUT_DIR
            / condition
            / class_name
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for index, image_path in enumerate(
            images,
            start=1
        ):

            image = load_image(
                image_path
            )

            image = resize_image(
                image
            )

            output_image = apply_condition(
                image,
                condition
            )

            output_name = (
                f"{prefix}_{index:04d}.jpg"
            )

            output_path = (
                output_folder
                / output_name
            )

            output_image.save(
                output_path,
                "JPEG",
                quality=95
            )

        print(
            f"{condition:<10}: "
            f"{len(images)} images"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    print("=" * 60)
    print("WEATHER AUGMENTATION")
    print("=" * 60)

    print(
        f"\nSource: {SOURCE_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for class_name, prefix in CLASSES.items():

        process_class(
            class_name,
            prefix
        )

    print("\n" + "=" * 60)
    print("WEATHER DATASET CREATED")
    print("=" * 60)

    print("\nConditions:")
    print("1. Daylight")
    print("2. Rainy")
    print("3. Snow")
    print("4. Night")

    print("\nExpected dataset:")
    print("Pedestrians : 170 × 4 = 680")
    print("Road Signs  : 150 × 4 = 600")
    print("Total       : 1280 images")

    print(
        "\nSaved to:"
        "\ndata/dataset_weather/"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()