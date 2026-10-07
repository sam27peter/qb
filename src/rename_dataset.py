from pathlib import Path
import shutil


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "data" / "dataset"
OUTPUT_DIR = BASE_DIR / "data" / "dataset_crct"


# ============================================================
# CLASS SETTINGS
# ============================================================

CLASS_SETTINGS = {
    "pedestrians": "pedestrian",
    "road_signs": "road_sign",
}


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# RENAME AND COPY
# ============================================================

def process_class(folder_name, prefix):

    source_folder = SOURCE_DIR / folder_name
    output_folder = OUTPUT_DIR / folder_name

    if not source_folder.exists():
        raise FileNotFoundError(
            f"Source folder not found: {source_folder}"
        )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        [
            file
            for file in source_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in VALID_EXTENSIONS
        ]
    )

    print(f"\n{folder_name}")
    print("-" * 40)
    print(f"Found images: {len(images)}")

    for index, image_path in enumerate(images, start=1):

        # Keep original extension
        extension = image_path.suffix.lower()

        new_name = f"{prefix}_{index:04d}{extension}"

        destination = output_folder / new_name

        shutil.copy2(
            image_path,
            destination
        )

    print(f"Copied images: {len(images)}")
    print(f"Output: {output_folder}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATING CORRECTLY NAMED DATASET")
    print("=" * 60)

    # Create main output folder
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Process both classes
    for folder_name, prefix in CLASS_SETTINGS.items():

        process_class(
            folder_name,
            prefix
        )

    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)

    print("\nFinal structure:")
    print("data/dataset_crct/pedestrians/")
    print("data/dataset_crct/road_signs/")

    print("\nExpected counts:")
    print("Pedestrians : 170")
    print("Road Signs  : 150")
    print("Total       : 320")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()