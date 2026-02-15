"""
Prepare YOLOv8-seg dataset from raw_data (Supervisely-style JSON + images).
Converts bitmap annotations to YOLO-seg polygon .txt, splits 80/20 train/val,
and copies files to datasets/train and datasets/val.
"""

import base64
import json
import random
import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_IMAGES = PROJECT_ROOT / "raw_data" / "archive" / "dataset" / "original_data" / "images"
RAW_ANNOTATIONS = PROJECT_ROOT / "raw_data" / "archive" / "dataset" / "original_data" / "annotations"
DATASETS = PROJECT_ROOT / "datasets"
TRAIN_IMAGES = DATASETS / "train" / "images"
TRAIN_LABELS = DATASETS / "train" / "labels"
VAL_IMAGES = DATASETS / "val" / "images"
VAL_LABELS = DATASETS / "val" / "labels"

# Class mapping: trash -> Debris (0), anything else -> Marine_Life (1)
CLASS_MAP = {"trash": 0, "debris": 0, "marine_life": 1, "fish": 1}
DEFAULT_CLASS = 0  # Debris for unknown classes

TRAIN_FRAC = 0.8
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def decode_bitmap_data(b64_data: str) -> np.ndarray:
    """Decode Supervisely bitmap base64 data to binary mask (0/255)."""
    raw = base64.b64decode(b64_data)
    # Try zlib decompress (common in Supervisely)
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass
    # Decode as image (PNG)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return mask


def mask_to_polygon(mask: np.ndarray, origin_x: int, origin_y: int, img_w: int, img_h: int) -> list[list[float]]:
    """Convert binary mask to list of normalized polygon(s) for YOLO-seg (each poly: [x1,y1,x2,y2,...])."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < 9:  # skip tiny contours
            continue
        # Contour in full image coords (mask is placed at origin)
        pts = c.reshape(-1, 2).astype(np.float32)
        pts[:, 0] += origin_x
        pts[:, 1] += origin_y
        # Normalize to 0-1
        pts[:, 0] /= img_w
        pts[:, 1] /= img_h
        pts = np.clip(pts, 0, 1)
        if len(pts) < 3:
            continue
        polygons.append(pts.flatten().tolist())
    return polygons


def class_title_to_id(title: str) -> int:
    key = (title or "").strip().lower().replace(" ", "_")
    return CLASS_MAP.get(key, DEFAULT_CLASS)


def process_annotation(json_path: Path, img_w: int, img_h: int) -> list[tuple[int, list[float]]]:
    """Read Supervisely JSON; return list of (class_id, polygon) for YOLO-seg."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    results = []
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue
        bmp = obj.get("bitmap")
        if not bmp:
            continue
        origin = bmp.get("origin", [0, 0])
        ox, oy = int(origin[0]), int(origin[1])
        mask = decode_bitmap_data(bmp.get("data", ""))
        if mask is None:
            continue
        polys = mask_to_polygon(mask, ox, oy, img_w, img_h)
        cid = class_title_to_id(obj.get("classTitle", ""))
        for poly in polys:
            results.append((cid, poly))
    return results


def polygon_line(cid: int, poly: list[float]) -> str:
    """One line for YOLO-seg .txt: class_id x1 y1 x2 y2 ..."""
    return f"{cid} " + " ".join(f"{x:.6f}" for x in poly)


def main():
    if not RAW_IMAGES.is_dir() or not RAW_ANNOTATIONS.is_dir():
        print(f"Missing raw_data: images={RAW_IMAGES}, annotations={RAW_ANNOTATIONS}")
        return 1

    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
    VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    VAL_LABELS.mkdir(parents=True, exist_ok=True)

    # Collect (image_path, annotation_path) for images that have JSON
    pairs = []
    for img_path in RAW_IMAGES.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        ann_path = RAW_ANNOTATIONS / f"{img_path.name}.json"
        if not ann_path.is_file():
            continue
        pairs.append((img_path, ann_path))

    if not pairs:
        print("No (image, annotation) pairs found.")
        return 1

    random.seed(42)
    random.shuffle(pairs)
    n_train = max(1, int(len(pairs) * TRAIN_FRAC))
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    def process_split(split_pairs: list, imgs_dir: Path, labels_dir: Path, split_name: str):
        for img_path, ann_path in split_pairs:
            stem = img_path.stem
            ext = img_path.suffix
            # Get image size for normalization
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
            lines = process_annotation(ann_path, w, h)
            dest_img = imgs_dir / f"{stem}{ext}"
            dest_txt = labels_dir / f"{stem}.txt"
            if not lines:
                dest_txt.write_text("", encoding="utf-8")
            else:
                dest_txt.write_text("\n".join(polygon_line(cid, p) for cid, p in lines), encoding="utf-8")
            shutil.copy2(img_path, dest_img)

        print(f"  {split_name}: {len(split_pairs)} images -> {imgs_dir}")

    print("Preparing dataset...")
    process_split(train_pairs, TRAIN_IMAGES, TRAIN_LABELS, "train")
    process_split(val_pairs, VAL_IMAGES, VAL_LABELS, "val")
    print("Done. You can run: py train.py")
    return 0


if __name__ == "__main__":
    exit(main())
