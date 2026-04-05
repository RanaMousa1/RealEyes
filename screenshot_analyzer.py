"""
screenshot_analyzer.py
-----------------------
Analyzes a screenshot and returns:
  - Extracted text  (if any)
  - Cropped image   (if any)

No AI API needed. Uses EasyOCR + OpenCV only.

Install dependencies:
    pip install easyocr opencv-python pillow numpy

Usage as a module:
    from screenshot_analyzer import analyze
    result = analyze("path/to/screenshot.png")

Usage from command line:
    python screenshot_analyzer.py path/to/screenshot.png
"""

import sys
import cv2
import numpy as np
import easyocr
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog


# Windows only: uncomment and set your tesseract path if needed
# (No longer needed with EasyOCR)


# ── Main function ─────────────────────────────────────────────
def analyze(image_path: str, output_dir: str = None) -> dict:
    """
    Analyze a screenshot and extract text and/or embedded image.

    Args:
        image_path : path to input screenshot
        output_dir : where to save cropped image (default: same folder as input)

    Returns:
        {
            "content_type" : "text_only" | "image_only" | "text_and_image",
            "text"         : str | None,
            "image_path"   : str | None
        }
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(output_dir) if output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img_bgr.shape[:2]

    # Step 1: extract text via OCR
    text = _extract_text(img_bgr)
    has_text = bool(text and text.strip())

    # Step 2: detect embedded photo/image region
    image_region = _detect_image_region(img_bgr)
    has_image = image_region is not None

    # Step 3: decide content type
    if has_text and has_image:
        content_type = "text_and_image"
    elif has_text:
        content_type = "text_only"
    else:
        content_type = "image_only"
        image_region = (0, 0, w, h)   # whole screenshot is the image

    # Step 4: crop and save image region
    cropped_path = None
    if image_region:
        cropped_path = _crop_and_save(img_bgr, image_region, image_path, output_dir)

    return {
        "content_type" : content_type,
        "text"         : text.strip() if has_text else None,
        "image_path"   : str(cropped_path) if cropped_path else None,
    }


# ── Text extraction (OCR) ─────────────────────────────────────
def _extract_text(img_bgr: np.ndarray) -> str:
    """Run EasyOCR and return all detected text."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    reader = easyocr.Reader(['en', 'ar'])
    results = reader.readtext(rgb)
    text = ' '.join([res[1] for res in results])
    return text


# ── Image region detection ────────────────────────────────────
def _detect_image_region(img_bgr: np.ndarray):
    """
    Detect the largest embedded photo/image region inside the screenshot.

    Approach:
      - Use Canny edge detection to find visually dense (photo-like) areas
      - Dilate edges to form blobs
      - Pick the largest blob that looks like a photo (high color variance, big area)
      - Ignore regions that cover almost the entire screenshot

    Returns (x, y, w, h) or None.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Edge map
    edges = cv2.Canny(gray, 30, 100)

    # Dilate to merge nearby edges into blobs
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = (h * w) * 0.05   # must cover at least 5% of screenshot

    best       = None
    best_score = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue

        # Score = area × color variance  (photos are large AND colorful)
        roi        = img_bgr[y:y+ch, x:x+cw]
        color_var  = float(np.std(roi))
        score      = area * color_var

        if score > best_score:
            best_score = score
            best       = (x, y, cw, ch)

    if best is None:
        return None

    # If detected region is almost the whole image, it means the screenshot
    # itself IS the image — caller handles this case
    x, y, cw, ch = best
    coverage = (cw * ch) / (w * h)
    if coverage > 0.92:
        return None

    return best


# ── Crop & save ───────────────────────────────────────────────
def _crop_and_save(img_bgr: np.ndarray, region: tuple,
                   src_path: Path, output_dir: Path) -> Path:
    x, y, cw, ch = region
    cropped  = img_bgr[y:y+ch, x:x+cw]
    out_path = output_dir / f"{src_path.stem}_image.png"
    cv2.imwrite(str(out_path), cropped)
    return out_path


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    image_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")]
    )

    if not image_path:
        print("No file selected.")
        sys.exit(1)

    output_dir = filedialog.askdirectory(title="Select output directory (optional)")
    if not output_dir:
        output_dir = None

    print(f"\nAnalyzing: {image_path}")
    result = analyze(image_path, output_dir)

    print(f"\nContent type : {result['content_type']}")

    if result["text"]:
        print("\n── Extracted Text ──────────────────────────")
        print(result["text"])

    if result["image_path"]:
        print(f"\n── Extracted Image ─────────────────────────")
        print(f"Saved to: {result['image_path']}")
