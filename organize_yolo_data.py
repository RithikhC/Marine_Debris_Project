"""
Organize raw_data for YOLOv8 training.
Moves all images into an 'images' folder and all label files into a 'labels' folder.
Handles multiple datasets (e.g. TrashCan, Marine Life) and duplicate filenames.
"""

import argparse
from pathlib import Path
import shutil


# File extensions to treat as images and labels
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
LABEL_EXTENSIONS = {".txt", ".json"}


def get_unique_dest_path(dest_dir: Path, base_name: str, suffix: str) -> Path:
    """Return a path in dest_dir that does not overwrite existing files."""
    candidate = dest_dir / base_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    ext = candidate.suffix
    i = 1
    while True:
        candidate = dest_dir / f"{stem}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def path_to_safe_basename(relative_path: Path, new_ext: str = None) -> str:
    """Turn a relative path into a safe filename (e.g. for avoiding collisions)."""
    parts = relative_path.parts
    if new_ext is not None:
        stem = relative_path.stem
        return "_".join(parts[:-1] + (stem,)) + new_ext
    return "_".join(parts)


def organize_data(raw_data_root: Path, output_root: Path, move: bool = True) -> None:
    """
    Recursively find all images and label files under raw_data_root,
    and move/copy them into output_root/images and output_root/labels.
    Uses path-based unique names to avoid overwriting when same filename
    appears in different subfolders (e.g. train/img1.jpg from two datasets).
    """
    raw_data_root = raw_data_root.resolve()
    output_root = output_root.resolve()

    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    op = shutil.move if move else shutil.copy2
    op_name = "Moving" if move else "Copying"

    image_count = 0
    label_count = 0

    for path in raw_data_root.rglob("*"):
        if not path.is_file():
            continue

        try:
            rel = path.relative_to(raw_data_root)
        except ValueError:
            continue

        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            # Use path-based name to avoid collisions (e.g. archive_dataset_original_data_images_foo.jpg)
            safe_name = path_to_safe_basename(rel, suffix)
            dest = get_unique_dest_path(images_dir, safe_name, suffix)
            op(str(path), str(dest))
            image_count += 1
            if image_count <= 5 or image_count % 2000 == 0:
                print(f"  {op_name} image: {rel} -> {dest.name}")

        elif suffix in LABEL_EXTENSIONS:
            safe_name = path_to_safe_basename(rel, suffix)
            dest = get_unique_dest_path(labels_dir, safe_name, suffix)
            op(str(path), str(dest))
            label_count += 1
            if label_count <= 5 or label_count % 2000 == 0:
                print(f"  {op_name} label: {rel} -> {dest.name}")

    print(f"\nDone. {op_name.lower()} {image_count} images and {label_count} label files.")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize raw_data into images/ and labels/ for YOLOv8 training."
    )
    parser.add_argument(
        "raw_data",
        nargs="?",
        type=Path,
        default=None,
        help="Path to raw_data folder (default: 'raw_data' next to this script)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output folder for images/ and labels/ (default: same as parent of raw_data)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving (keeps raw_data unchanged)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    raw_data_root = args.raw_data or (script_dir / "raw_data")
    if not raw_data_root.exists():
        print(f"Error: raw_data path does not exist: {raw_data_root}")
        return 1

    output_root = args.output or raw_data_root.parent
    print(f"Source: {raw_data_root}")
    print(f"Output: {output_root} (images/, labels/)")
    print()

    organize_data(raw_data_root, output_root, move=not args.copy)
    return 0


if __name__ == "__main__":
    exit(main())
