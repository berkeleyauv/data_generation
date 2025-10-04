# Data Generation
Scripts for generating synthetic datasets for deep learning. Includes random augmentation and supports dynamic overlays, automatic class mapping, and even distribution of target images across all backgrounds. 

## Dataset location
This code expects:
DATA_ROOT/
├── backgrounds/        # Folder containing background images (.jpg/.png)
├── target/             # Folder containing target images (.jpg/.png)
└── output/             # Generated composites and labels will be saved here

## Usage
Run from the project root:

python generate.py \
  --backgrounds_dir DATA_ROOT/backgrounds \
  --targets_dir DATA_ROOT/target \
  --output_dir DATA_ROOT/output \
  --overlay_func compositor.overlay.overlay_gate.paste_overlay

(currently provides only one overlay function: overlay_gate. can replace with more kinds in the future)
