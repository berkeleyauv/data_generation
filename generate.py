import os
import argparse
from PIL import Image
from compositor.overlay.overlay_multiple import paste_multiple_overlay

def load_images_from_folder(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, f)
            try:
                img = Image.open(path)
                files.append(img.copy())
                img.close()
            except:
                print(f"Could not load image: {path}")
    return files

def main(backgrounds_dir, real_targets_dir, fake_targets_dir, output_img_dir, output_yolo_dir, max_attempts, num_backgrounds=None):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_yolo_dir, exist_ok=True)

    backgrounds = load_images_from_folder(backgrounds_dir)
    real_targets = load_images_from_folder(real_targets_dir)

    fake_targets = []
    if fake_targets_dir:
        fake_targets = load_images_from_folder(fake_targets_dir)

    if num_backgrounds is not None:
        backgrounds = backgrounds[:num_backgrounds]

    for i, bg in enumerate(backgrounds):
        composite, labels = paste_multiple_overlay(
            bg, real_targets, fake_targets, max_attempts=max_attempts
        )

        img_name = f"img_{i:05d}.jpg"
        label_name = f"img_{i:05d}.txt"

        img_path = os.path.join(output_img_dir, img_name)
        yolo_path = os.path.join(output_yolo_dir, label_name)

        composite.save(img_path, quality=95)

        with open(yolo_path, "w") as f:
            for lb in labels:
                class_id = lb["class_id"]
                x, y, w, h = lb["bbox"]
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"Generated {img_name} + {label_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backgrounds_dir", required=True)
    parser.add_argument("--real_targets_dir", required=True)
    parser.add_argument("--fake_targets_dir", default=None, help="Folder with fake target images (optional)")

    parser.add_argument("--output_img_dir", required=True)
    parser.add_argument("--output_yolo_dir", required=True)

    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--num_backgrounds", type=int, default=None, help="Number of backgrounds to process")

    args = parser.parse_args()

    main(
        args.backgrounds_dir,
        args.real_targets_dir,
        args.fake_targets_dir,
        args.output_img_dir,
        args.output_yolo_dir,
        args.max_attempts,
        args.num_backgrounds
    )
