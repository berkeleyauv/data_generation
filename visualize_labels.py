import os
import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR  = "output_images"         
LABEL_DIR  = "output_yolo"        
OUTPUT_DIR = "visualized"  

# Optional: map class IDs to names, e.g. {0: "cat", 1: "dog"}
CLASS_NAMES = {}

#Drawing Settings
BOX_THICKNESS  = 2
FONT_SCALE     = 0.6
FONT_THICKNESS = 1

# If True, each class gets a unique color; if False, all boxes are green
COLORIZE_BY_CLASS = True
# ─────────────────────────────────────────────────────────────────────────────

# color per class/image
def class_color(class_id: int):
    np.random.seed(class_id + 42)
    return tuple(int(c) for c in np.random.randint(50, 230, 3))


def draw_yolo_boxes(image_path: Path, label_path: Path, output_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  [WARN] Could not read image: {image_path}")
        return

    h, w = img.shape[:2]

    if not label_path.exists():
        print(f"  [WARN] No label file for: {image_path.name}")
        # Save image as-is so output set is complete
        cv2.imwrite(str(output_path), img)
        return

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # Convert normalized → pixel coords
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = class_color(class_id) if COLORIZE_BY_CLASS else (0, 255, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        label = CLASS_NAMES.get(class_id, str(class_id))
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        # Background rectangle for text
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
            (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA
        )

    cv2.imwrite(str(output_path), img)


def main():
    image_dir  = Path(IMAGE_DIR)
    label_dir  = Path(LABEL_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]

    if not image_files:
        print(f"No images found in '{IMAGE_DIR}'. Check your IMAGE_DIR setting.")
        return

    print(f"Found {len(image_files)} images. Processing...")

    for i, img_path in enumerate(sorted(image_files), 1):
        label_path  = label_dir  / (img_path.stem + ".txt")
        output_path = output_dir / img_path.name
        draw_yolo_boxes(img_path, label_path, output_path)
        if i % 500 == 0 or i == len(image_files):
            print(f"  {i}/{len(image_files)} done")

    print(f"\nDone! Visualized images saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()