from load_env import (
    BACKGROUNDS_DIR, TARGET_ASSETS_DIR, OUTPUT_DIR
)

import os
import random
import argparse
from PIL import Image
import numpy as np
from datetime import datetime

from compositor.augment import augment 
from compositor.overlay import overlay_gate

TARGET_PATHS = {
    "reef_shark": os.path.join(TARGET_ASSETS_DIR, "Task01_ReefShark.png"),
    "sawfish": os.path.join(TARGET_ASSETS_DIR, "Task01_Sawfish.png")
}

CLASS_MAP = {
    "reef_shark": 0,
    "sawfish": 1
}

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, f'{timestamp}/images')
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DIR, f'{timestamp}/labels')

def get_random_background(bg_files):
    bg_path = random.choice(bg_files)
    return Image.open(bg_path).convert("RGB")

def get_random_target():
    class_name = random.choice(["reef_shark", "sawfish"])
    class_id = CLASS_MAP[class_name]
    target_path = TARGET_PATHS[class_name]
    target_img = Image.open(target_path).convert("RGBA")
    return target_img, class_id

def save_label(label_dict, label_path):
    class_id = label_dict['class_id']
    xc, yc, w, h = label_dict['bbox']
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {xc} {yc} {w} {h}\n")

def main(num_samples):
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    bg_files = [os.path.join(BACKGROUNDS_DIR, f) for f in os.listdir(BACKGROUNDS_DIR)]

    i = 0
    while i < num_samples:
        bg = get_random_background(bg_files)
        target_img, class_id = get_random_target()

        # augment target image
        target_img_aug = augment(target_img)

        # overlay onto random background
        composite, label = overlay_gate.paste_overlay(bg, target_img_aug, class_id)

        # Handle cases where overlay didn't fit
        if composite is None:
            continue

        # Save output image and label
        image_filename = f"sample_{i:05d}.jpg"
        label_filename = f"sample_{i:05d}.txt"

        composite.save(os.path.join(OUTPUT_IMAGES_DIR, image_filename))
        save_label(label, os.path.join(OUTPUT_LABELS_DIR, label_filename))

        print(f"[{i}] Saved {image_filename}")
        i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    main(args.num_samples)
