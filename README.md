# Data Generation
Scripts for generating synthetic datasets for deep learning. Includes random augmentation and supports dynamic overlays, automatic class mapping, and even distribution of target images across all backgrounds. 

## Directory Setup
This code expects:
```bash
DATA_ROOT/
├── backgrounds/        # Folder containing background images (.jpg/.png)
├── targets_original/   # Folder containing real target images (.jpg/.png)
├── targets_fake/       # Folder containing fake target images (.jpg/.png)
```
## Usage
Run from the project root:

without fake images (generates 10):

```bash
python generate.py \
  --backgrounds_dir backgrounds \
  --real_targets_dir targets_original \
  --output_img_dir output_images \
  --output_yolo_dir output_yolo \
  --max_attempts 20 \
  --num_backgrounds 10
```

to have multiple target images placed on your image, instead use generate_multiple.py.

```bash
python generate_multiple.py \
  --backgrounds_dir backgrounds \
  --real_targets_dir targets_original \
  --output_img_dir output_images \
  --output_yolo_dir output_yolo \
  --max_attempts 20 \
  --num_backgrounds 10
```
with fake images (adds a single fake image (randomly chosen from a set of fake images) along with real target images)

```bash
python generate.py \
  --backgrounds_dir backgrounds \
  --real_targets_dir targets_original \
  --fake_targets_dir targets_fake \
  --output_img_dir output_images \
  --output_yolo_dir output_yolo \
  --max_attempts 20 \
  --num_backgrounds 10
```
torpedo_overlay.py run command
(must already have output folders WITHOUT fake images)

```bash
python -m compositor.overlay.overlay_torpedo \
  --images_dir output_images \
  --yolo_dir output_yolo \
  --output_dir torpedo_images \
  --output_yolo_dir torpedo_yolo
```

To visualize labels: 
python visualize_labels.py