"""
Smoke test for the full data generation pipeline.

Uses real fixture images (tiny) to run the pipeline end-to-end and validate outputs.

Run with:
    pytest tests/test_smoke.py -v
"""

import os
import pytest
from PIL import Image
from generate import main

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
BACKGROUNDS_DIR = os.path.join(FIXTURES_DIR, "backgrounds")
TARGETS_DIR = os.path.join(FIXTURES_DIR, "targets")


@pytest.fixture
def output_dirs(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()
    return str(img_dir), str(lbl_dir)


class TestSmokePipeline:
    def test_pipeline_runs_without_crashing(self, output_dirs):
        """Full pipeline completes on real fixture images without raising."""
        img_dir, lbl_dir = output_dirs
        main(
            backgrounds_dir=BACKGROUNDS_DIR,
            real_targets_dir=TARGETS_DIR,
            fake_targets_dir=None,
            output_img_dir=img_dir,
            output_yolo_dir=lbl_dir,
            max_attempts=20,
        )

    def test_correct_number_of_outputs(self, output_dirs):
        """One image + one label file per background."""
        img_dir, lbl_dir = output_dirs
        main(
            backgrounds_dir=BACKGROUNDS_DIR,
            real_targets_dir=TARGETS_DIR,
            fake_targets_dir=None,
            output_img_dir=img_dir,
            output_yolo_dir=lbl_dir,
            max_attempts=20,
        )
        images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        labels = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
        num_backgrounds = len([f for f in os.listdir(BACKGROUNDS_DIR)])
        assert len(images) == num_backgrounds
        assert len(labels) == num_backgrounds

    def test_output_images_are_valid(self, output_dirs):
        """Every output image can be opened and is RGB."""
        img_dir, lbl_dir = output_dirs
        main(
            backgrounds_dir=BACKGROUNDS_DIR,
            real_targets_dir=TARGETS_DIR,
            fake_targets_dir=None,
            output_img_dir=img_dir,
            output_yolo_dir=lbl_dir,
            max_attempts=20,
        )
        for fname in os.listdir(img_dir):
            if fname.endswith(".jpg"):
                img = Image.open(os.path.join(img_dir, fname))
                assert img.mode == "RGB", f"{fname} is not RGB"
                assert img.size == (640, 480), f"{fname} has wrong size"

    def test_label_files_have_valid_yolo_format(self, output_dirs):
        """Every non-empty label file contains valid YOLO formatted lines."""
        img_dir, lbl_dir = output_dirs
        main(
            backgrounds_dir=BACKGROUNDS_DIR,
            real_targets_dir=TARGETS_DIR,
            fake_targets_dir=None,
            output_img_dir=img_dir,
            output_yolo_dir=lbl_dir,
            max_attempts=20,
        )
        for fname in os.listdir(lbl_dir):
            fpath = os.path.join(lbl_dir, fname)
            content = open(fpath).read().strip()
            if not content:
                continue  # empty label is valid (no placement)

            for line in content.splitlines():
                parts = line.strip().split()
                assert len(parts) == 5, f"{fname}: expected 5 values, got {len(parts)}"

                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])

                assert class_id >= 1, f"{fname}: class_id must be >= 1"
                assert 0.0 <= x <= 1.0, f"{fname}: x_center out of range: {x}"
                assert 0.0 <= y <= 1.0, f"{fname}: y_center out of range: {y}"
                assert 0.0 < w <= 1.0,  f"{fname}: width out of range: {w}"
                assert 0.0 < h <= 1.0,  f"{fname}: height out of range: {h}"
                assert w > 0 and h > 0,  f"{fname}: zero-area bbox"

                # No NaNs
                for val in [x, y, w, h]:
                    assert val == val, f"{fname}: NaN found in bbox"

    def test_num_backgrounds_limit(self, output_dirs):
        """num_backgrounds parameter correctly limits output count."""
        img_dir, lbl_dir = output_dirs
        main(
            backgrounds_dir=BACKGROUNDS_DIR,
            real_targets_dir=TARGETS_DIR,
            fake_targets_dir=None,
            output_img_dir=img_dir,
            output_yolo_dir=lbl_dir,
            max_attempts=20,
            num_backgrounds=1,
        )
        images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        assert len(images) == 1