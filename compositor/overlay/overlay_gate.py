from PIL import Image
import random
import numpy as np
import cv2

def random_perspective_transform(img_np, max_offset=0.2):
    h, w = img_np.shape[:2]
    dx = int(w * max_offset)
    dy = int(h * max_offset)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, dx), random.randint(0, dy)],
        [w - random.randint(0, dx), random.randint(0, dy)],
        [w - random.randint(0, dx), h - random.randint(0, dy)],
        [random.randint(0, dx), h - random.randint(0, dy)]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_np, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return warped

def paste_overlay(
    background: Image.Image,
    overlay: Image.Image,
    class_id: int,
    scale_range=(0.1, 0.7),
    rotation_range=(-60, 60),
    perspective_offset=0.15,
):
    bg = background.copy().convert("RGBA")
    ov = overlay.convert("RGBA")

    bg_w, bg_h = bg.size

    # Randomly scale overlay
    scale = random.uniform(*scale_range)
    new_w = int(bg_w * scale)
    aspect = ov.width / ov.height
    new_h = int(new_w / aspect)
    ov = ov.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Random rotation
    angle = random.uniform(*rotation_range)
    ov = ov.rotate(angle, expand=True)

    ov_np = np.array(ov)
    ov_np = random_perspective_transform(ov_np, max_offset=perspective_offset)
    ov = Image.fromarray(ov_np)

    new_w, new_h = ov.size
    if new_w >= bg_w or new_h >= bg_h:
        return None, None

    # Random position
    x_min = random.randint(0, bg_w - new_w)
    y_min = random.randint(0, bg_h - new_h)

    # Paste using alpha
    bg.paste(ov, (x_min, y_min), ov)

    # YOLO bounding box
    x_center = (x_min + new_w / 2) / bg_w
    y_center = (y_min + new_h / 2) / bg_h
    w_norm = new_w / bg_w
    h_norm = new_h / bg_h

    label = {
        "class_id": class_id,
        "bbox": (x_center, y_center, w_norm, h_norm)
    }

    return bg.convert("RGB"), label

# Testing the function
if __name__ == '__main__':
    background = Image.open('/Users/kvaikunthan/Desktop/data_generation/test_data/tacos.jpg').convert("RGB")
    overlay_img = Image.open('/Users/kvaikunthan/Desktop/data_generation/test_data/Task01_Sawfish.png').convert("RGBA")
    img, label = paste_overlay(background, overlay_img, 0, scale_range=(0.1, 0.9), perspective_offset=0.2)
    if img:
        composite_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        print(label)
        cv2.imshow('Augmented Composite', composite_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Overlay skipped due to size mismatch.")
