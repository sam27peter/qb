from pathlib import Path
from PIL import Image
from collections import Counter


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "dataset_crct"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# INSPECT IMAGE SIZES
# ============================================================

def inspect_class(class_name):

    folder = DATASET_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    images = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ]

    size_counts = Counter()

    min_width = float("inf")
    max_width = 0

    min_height = float("inf")
    max_height = 0

    valid_count = 0
    invalid_count = 0

    for image_path in images:

        try:
            with Image.open(image_path) as image:

                width, height = image.size

                size_counts[(width, height)] += 1

                min_width = min(min_width, width)
                max_width = max(max_width, width)

                min_height = min(min_height, height)
                max_height = max(max_height, height)

                valid_count += 1

        except Exception:
            invalid_count += 1

    print(f"\n{class_name}")
    print("=" * 50)

    print(f"Total images     : {len(images)}")
    print(f"Valid images     : {valid_count}")
    print(f"Invalid images   : {invalid_count}")

    if valid_count == 0:
        return

    print("\nWidth:")
    print(f"Minimum          : {min_width}")
    print(f"Maximum          : {max_width}")

    print("\nHeight:")
    print(f"Minimum          : {min_height}")
    print(f"Maximum          : {max_height}")

    print("\nMost common dimensions:")

    for (width, height), count in size_counts.most_common(10):

        print(
            f"{width} × {height}"
            f"  →  {count} images"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMAGE DIMENSION INSPECTION")
    print("=" * 60)

    print(f"\nDataset: {DATASET_DIR}")

    inspect_class("pedestrians")
    inspect_class("road_signs")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()