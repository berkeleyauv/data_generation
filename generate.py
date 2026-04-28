import os
import argparse
from PIL import Image
import random
from compositor.overlay.overlay_single import paste_single_overlay

#LazyLoading: Loads paths instead of all the images at once so that the system doesn't crash
 # useful because strings hold less memory than 
def load_paths_from_folder(folder):
    paths = []
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")): 
            paths.append(os.path.join(folder, f)) #attached any file path that is an image within the folder. 
    return paths

def main(backgrounds_dir, real_targets_dir, fake_targets_dir, output_img_dir, output_yolo_dir, max_attempts, num_backgrounds=None):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_yolo_dir, exist_ok=True)

    background_paths = load_paths_from_folder(backgrounds_dir)
    real_target_paths = load_paths_from_folder(real_targets_dir)

    if num_backgrounds is not None:
        background_paths = background_paths[:num_backgrounds]

    for i, bg_path in enumerate(background_paths):
        #chooses a random index from the target images to use on each background
        target_idx = random.randint(0, len(real_target_paths) - 1)
        class_id = target_idx + 1

        #Adjusted opening method to use paths instead of the actual image. 
        with Image.open(bg_path) as bg:
            with Image.open(real_target_paths[target_idx]) as target:
                composite, labels = paste_single_overlay(
                    bg.copy(), target.copy(), class_id=class_id, max_attempts=max_attempts
                )

        img_name = f"img_{i:05d}.jpg"
        label_name = f"img_{i:05d}.txt"

        img_path = os.path.join(output_img_dir, img_name)
        yolo_path = os.path.join(output_yolo_dir, label_name)

        composite.save(img_path, quality=95)

        bbox = labels.get("bbox")
        if not bbox:
            print(f"Warning: no placement for {img_name}, saving empty label")
            open(yolo_path, "w").close()
            continue

        with open(yolo_path, "w") as f:
            x, y, w, h = bbox
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        print(f"Generated {img_name} + {label_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backgrounds_dir", required=True)
    parser.add_argument("--real_targets_dir", required=True)
    parser.add_argument("--fake_targets_dir", default=None)
    parser.add_argument("--output_img_dir", required=True)
    parser.add_argument("--output_yolo_dir", required=True)
    parser.add_argument("--max_attempts", type=int, default=20)
    parser.add_argument("--num_backgrounds", type=int, default=None)

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