import os
import random
import argparse
from PIL import Image
from datetime import datetime

from compositor.augment import augment
from compositor.overlay import overlay_gate

CLASS_MAP = {
    "reefshark": 0,
    "sawfish": 1
}

def get_random_background(bg_files):
    bg_path = random.choice(bg_files)
    return Image.open(bg_path).convert("RGB")

def get_random_target(target_paths):
    class_name = random.choice(list(target_paths.keys()))
    class_id = CLASS_MAP[class_name]
    target_path = target_paths[class_name]
    target_img = Image.open(target_path).convert("RGBA")
    return target_img, class_id

def save_label(label_dict, label_path):
    class_id = label_dict['class_id']
    xc, yc, w, h = label_dict['bbox']
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {xc} {yc} {w} {h}\n")

def main(backgrounds_dir, targets_dir, output_dir, num_samples):
    # Build target paths dynamically from the folder
    target_files = [f for f in os.listdir(targets_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    target_paths = {os.path.splitext(f)[0]: os.path.join(targets_dir, f) for f in target_files}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_images_dir = os.path.join(output_dir, f'{timestamp}/images')
    output_labels_dir = os.path.join(output_dir, f'{timestamp}/labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    bg_files = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    i = 0
    while i < num_samples:
        bg = get_random_background(bg_files)
        target_img, class_id = get_random_target(target_paths)

        # augment target image
        target_img_aug = augment(target_img)

        # overlay onto random background
        composite, label = overlay_gate.paste_overlay(bg, target_img_aug, class_id)

        if composite is None:
            continue

        # Save output image and label
        image_filename = f"sample_{i:05d}.jpg"
        label_filename = f"sample_{i:05d}.txt"
        composite.save(os.path.join(output_images_dir, image_filename))
        save_label(label, os.path.join(output_labels_dir, label_filename))

        print(f"[{i}] Saved {image_filename}")
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate augmented overlay dataset")
    parser.add_argument('--backgrounds_dir', type=str, required=True, help="Folder containing background images")
    parser.add_argument('--targets_dir', type=str, required=True, help="Folder containing target images")
    parser.add_argument('--output_dir', type=str, required=True, help="Folder to save output images and labels")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()

    main(args.backgrounds_dir, args.targets_dir, args.output_dir, args.num_samples)
