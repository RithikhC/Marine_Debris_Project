"""
Preprocess training images: remove text artifacts (timestamps) via inpainting,
then sharpen object boundaries using Canny edge enhancement.
Overwrites images in datasets/train/images.
"""

import cv2
import numpy as np
from pathlib import Path


# Where to look for text (timestamps): fraction of image height from bottom, width from sides
BOTTOM_FRAC = 0.25
SIDE_FRAC = 0.20
# Min contour area (pixels) to treat as text; smaller = text
TEXT_CONTOUR_AREA_MAX = 3000
# Inpaint and Canny params
INPAINT_RADIUS = 3
CANNY_LOW, CANNY_HIGH = 50, 150
EDGE_STRENGTH = 0.3  # blend Canny edges back into image (0–1)


def build_text_mask(bgr: np.ndarray) -> np.ndarray:
    """Build a mask of likely text regions (e.g. timestamps) in bottom/corners."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ROI: bottom strip and side margins where timestamps usually are
    bottom_y = int(h * (1 - BOTTOM_FRAC))
    side_w = int(w * SIDE_FRAC)
    roi = gray[bottom_y:, :].copy()
    left_roi = gray[:, :side_w].copy()
    right_roi = gray[:, -side_w:].copy()

    mask = np.zeros_like(gray, dtype=np.uint8)

    def add_text_regions(region: np.ndarray, out_mask: np.ndarray, y_offset: int = 0, x_offset: int = 0) -> None:
        # High-contrast regions: invert and use both light-on-dark and dark-on-light
        _, thresh_dark = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, thresh_light = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(thresh_dark, thresh_light)
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        # Find contours; small ones are likely text
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < TEXT_CONTOUR_AREA_MAX and cv2.contourArea(c) > 20:
                # Draw filled contour into output mask at correct offset
                c_shift = c + np.array([x_offset, y_offset])
                cv2.drawContours(out_mask, [c_shift], -1, 255, -1)

    add_text_regions(roi, mask, y_offset=bottom_y, x_offset=0)
    add_text_regions(left_roi, mask, y_offset=0, x_offset=0)
    add_text_regions(right_roi, mask, y_offset=0, x_offset=w - side_w)

    # If we found almost no text, fall back to masking bottom corners (common timestamp spots)
    if np.sum(mask > 0) < 100:
        corner_h, corner_w = int(h * 0.15), int(w * 0.2)
        mask[-(corner_h + 1) :, : (corner_w + 1)] = 255
        mask[-(corner_h + 1) :, -(corner_w + 1) :] = 255

    # Slight dilate so inpainting covers edges of text
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel)
    return mask


def sharpen_with_canny(bgr: np.ndarray) -> np.ndarray:
    """Sharpen object boundaries by blending Canny edges back into the image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Blend: image + strength * edges (so boundaries get a bright boost)
    sharpened = cv2.addWeighted(bgr, 1.0, edges_bgr, EDGE_STRENGTH, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_image(path: Path) -> None:
    """Load image, remove text with inpainting, sharpen with Canny, save back."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"  Skip (not an image): {path.name}")
        return

    # 1) Mask text and inpaint
    text_mask = build_text_mask(img)
    if np.sum(text_mask > 0) > 0:
        img = cv2.inpaint(img, text_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

    # 2) Sharpen boundaries with Canny
    img = sharpen_with_canny(img)

    cv2.imwrite(str(path), img)


def main() -> None:
    images_dir = Path(__file__).resolve().parent / "datasets" / "train" / "images"
    if not images_dir.is_dir():
        print(f"Folder not found: {images_dir}")
        print("Create it and add images, or point preprocess.py at your images folder.")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        print(f"No images found in {images_dir}")
        return

    print(f"Processing {len(paths)} images in {images_dir}")
    for i, path in enumerate(paths, 1):
        process_image(path)
        if i % 50 == 0 or i == len(paths):
            print(f"  Done {i}/{len(paths)}")

    print("Done.")


if __name__ == "__main__":
    main()
