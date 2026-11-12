# Data Generation
Scripts for generating synthetic datasets for deep learning. Includes random augmentation and supports dynamic overlays, automatic class mapping, and even distribution of target images across all backgrounds.
**Step by Step demo at the bottom.** 


## Directory Setup
This code expects:
```bash
data_generation/        #repo root/project root
├── backgrounds/        # Folder containing background images (.jpg/.png)
├── targets_original/   # Folder containing real target images (.jpg/.png)
├── targets_fake/       # Folder containing fake target images (.jpg/.png)
├── compositor/ 
├── generate.py 
├── requirements.txt
...

```
**backgrounds** - In this you would put all the backgrounds you want to generate image target overlays over.


**targets_original** - Put all the cropped target images you want to lay on backgrounds here.


**targets_fake** - Put all the fake target images here.


## Usage
Run from the project root:

By default, `generate.py` writes YOLO pose labels: class ID, normalized bounding
box, then four corner keypoints in top-left, top-right, bottom-right, bottom-left
order. Each keypoint is stored as normalized `x y visibility`, with visibility
set to `2`, followed by normalized `[x, y, w, h]` boxes for the left post, right
post, and crossbar, in that order. Configure the training dataset with
`kpt_shape: [4, 3]`. Single-target generation requires a same-stem JSON sidecar
beside every target image, with normalized component boxes, for example:

```json
{
  "left_post": [0.05, 0.5, 0.08, 0.9],
  "right_post": [0.95, 0.5, 0.08, 0.9],
  "crossbar": [0.5, 0.08, 0.9, 0.08]
}
```

The appended component boxes make this a project-specific 29-value row, not a
standard YOLO pose row; use a matching label parser for training.

The `--multiple` mode writes regular YOLO detection boxes instead.


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


To place multiple target images on each background, add `--multiple`:


```bash
python generate.py \
 --backgrounds_dir backgrounds \
 --real_targets_dir targets_original \
 --output_img_dir output_images \
 --output_yolo_dir output_yolo \
 --max_attempts 20 \
 --num_backgrounds 10 \
 --multiple
```
With multiple mode, `--fake_targets_dir` optionally adds one randomly selected fake target per image. Single-target mode writes YOLO pose labels; multiple mode writes regular YOLO detection boxes.


```bash
python generate.py \
 --backgrounds_dir backgrounds \
 --real_targets_dir targets_original \
 --fake_targets_dir targets_fake \
 --output_img_dir output_images \
 --output_yolo_dir output_yolo \
 --max_attempts 20 \
 --num_backgrounds 10 \
 --multiple
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


Here's the shortened version with proper code blocks:

---

# Step by Step Demo

**Case:** Train a YOLOv8 model to detect a beachball in a pool from a camera feed.

## 1. Setup

```bash
git clone https://github.com/berkeleyauv/data_generation.git
```

- **Backgrounds** — pool/underwater images matching your deployment environment
- **Targets** — 50–200 images of the beachball across varied orientations, lighting, and scales
- **Fakes** *(optional)* — ~10–20 cropped images of beachball-like objects the model might confuse

## 2. Crop Your Targets (via Roboflow)

1. Upload target images to a [Roboflow](https://roboflow.com/) project
2. Manually label 10–20 images; auto-label the rest and review
3. Export a version using Roboflow's **crop tool**
4. Replace your `targets_original/` folder with the downloaded dataset

Repeat for fake targets if needed.

## 3. Generate Data

Ensure your directories are structured as:

```
targets_original/   # cropped target images
targets_fake/       # cropped fake images (optional)
backgrounds/        # background images
```

Then run:

```bash
python generate.py \
  --backgrounds_dir backgrounds \
  --real_targets_dir targets_original \
  --output_img_dir output_images \
  --output_yolo_dir output_yolo \
  --max_attempts 20 \
  --num_backgrounds 15000
```

- Add `--multiple` to `generate.py` to place multiple beachballs per image
- Adjust `--num_backgrounds` to change output count

## 4. Visualize Labels *(optional)*

```bash
python3 visualize_labels.py
```
## 5. Testing
In order to test if generated data produces the right boxes we use the following files to test our outputs: 
- tests/test_generate.py — unit tests for path loading and label file writing in the main pipeline
- tests/test_overlay_single.py — unit tests for the overlay logic including bbox math, normalization, color matching, and regression detection
- tests/test_smoke.py — end-to-end pipeline test using real fixture images to validate full output correctness
- tests/fixtures/ — tiny sample backgrounds and targets used by the smoke test
- pytest.ini — pytest configuration telling it to treat the repo root as the Python path
- conftest.py — empty file that marks the repo root as the pytest project root
- .github/workflows/ci.yml — GitHub Actions workflow that runs linting, formatting checks, and all tests on every push and pull request

Running tests by file:

``` bash
pytest tests/test_generate.py -v
pytest tests/test_overlay_single.py -v
pytest tests/test_smoke.py -v


```

Use the following command to run all of the tests at once: 

``` bash
pytest tests/ -v
```

If successful it should say "30 passed in 2.04s," there are 30 tests total.
