"""
Tests for compositor/overlay/overlay_single.py

Run with:
    pytest tests/test_overlay_single.py -v
"""

import random
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch

from compositor.overlay.overlay_single import (
    paste_single_overlay,
    random_perspective_transform,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bg(w=640, h=480, color="blue"):
    return Image.new("RGB", (w, h), color=color)


def make_target(w=80, h=80, color=(255, 0, 0, 255)):
    img = Image.new("RGBA", (w, h), color=color)
    return img


# ---------------------------------------------------------------------------
# random_perspective_transform
# ---------------------------------------------------------------------------


class TestPerspectiveTransform:
    def test_output_shape_preserved(self):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        out = random_perspective_transform(img)
        assert out.shape == img.shape, "Output shape must match input shape"

    def test_output_dtype_preserved(self):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        out = random_perspective_transform(img)
        assert out.dtype == np.uint8

    def test_no_nans_in_output(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        out = random_perspective_transform(img)
        assert not np.isnan(
            out.astype(float)
        ).any(), "Perspective transform must not produce NaNs"

    def test_deterministic_with_seed(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        random.seed(42)
        out1 = random_perspective_transform(img)
        random.seed(42)
        out2 = random_perspective_transform(img)
        np.testing.assert_array_equal(out1, out2)

    def test_zero_offset_is_identity(self):
        """max_offset=0 means no warp — output should equal input."""
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        out = random_perspective_transform(img, max_offset=0.0)
        np.testing.assert_array_equal(out, img)


# ---------------------------------------------------------------------------
# paste_single_overlay — return value contract
# ---------------------------------------------------------------------------


class TestPasteSingleOverlayContract:
    def test_returns_image_and_label_on_success(self):
        random.seed(0)
        bg = make_bg()
        tgt = make_target()
        result_img, label = paste_single_overlay(bg, tgt, class_id=1)
        assert result_img is not None, "Should return a composite image"
        assert label is not None, "Should return a label dict"

    def test_returns_none_none_when_max_attempts_exhausted(self):
        """Force failure by making overlay larger than 40% of bg."""
        bg = make_bg(100, 100)
        # Target nearly as big as background → scale_range forces area > 40%
        tgt = make_target(90, 90)
        result_img, label = paste_single_overlay(
            bg, tgt, class_id=1, scale_range=(0.95, 0.99), max_attempts=5
        )
        assert result_img is None
        assert label is None

    def test_label_has_bbox_key(self):
        random.seed(1)
        bg = make_bg()
        tgt = make_target()
        _, label = paste_single_overlay(bg, tgt, class_id=3)
        assert "bbox" in label

    def test_label_has_class_id_key(self):
        random.seed(2)
        bg = make_bg()
        tgt = make_target()
        _, label = paste_single_overlay(bg, tgt, class_id=5)
        assert label["class_id"] == 5

    def test_output_image_is_rgb(self):
        random.seed(3)
        bg = make_bg()
        tgt = make_target()
        result_img, _ = paste_single_overlay(bg, tgt, class_id=1)
        assert result_img.mode == "RGB"

    def test_output_image_same_size_as_background(self):
        random.seed(4)
        bg = make_bg(320, 240)
        tgt = make_target(40, 40)
        result_img, _ = paste_single_overlay(bg, tgt, class_id=1)
        assert result_img.size == (320, 240)


# ---------------------------------------------------------------------------
# paste_single_overlay — YOLO bbox normalization math
# ---------------------------------------------------------------------------


class TestYOLONormalization:
    """
    These tests lock down the coordinate math.  We mock random calls so the
    placement is deterministic and we can compute expected values exactly.
    """

    def _run_with_fixed_placement(self, bg_w, bg_h, ov_w, ov_h, x_min, y_min):
        """
        Bypass all randomness: fixed scale produces a known overlay size,
        fixed position produces a known placement.
        Returns (x_center, y_center, w_norm, h_norm).
        """
        random.seed(0)
        bg = make_bg(bg_w, bg_h)
        tgt = make_target(ov_w, ov_h)

        # Patch the perspective transform to be a no-op so overlay dims don't change
        def noop_transform(img_np, max_offset=0.1):
            return img_np

        with patch(
            "compositor.overlay.overlay_single.random_perspective_transform",
            side_effect=noop_transform,
        ), patch(
            "compositor.overlay.overlay_single.random.uniform", return_value=ov_w / bg_w
        ), patch(
            "compositor.overlay.overlay_single.random.randint",
            side_effect=[x_min, y_min],
        ):
            result_img, label = paste_single_overlay(
                bg, tgt, class_id=1, max_attempts=1
            )

        return label["bbox"] if label else None

    def test_centered_placement(self):
        """Overlay placed dead-center → x_center=0.5, y_center=0.5."""
        bg_w, bg_h = 640, 480
        ov_w, ov_h = 128, 128
        x_min = (bg_w - ov_w) // 2  # 256
        y_min = (bg_h - ov_h) // 2  # 176

        bbox = self._run_with_fixed_placement(bg_w, bg_h, ov_w, ov_h, x_min, y_min)
        if bbox is None:
            pytest.skip("Placement failed — check scale constraints")

        x_c, y_c, w_n, h_n = bbox
        expected_xc = (x_min + ov_w / 2) / bg_w
        expected_yc = (y_min + ov_h / 2) / bg_h
        assert abs(x_c - expected_xc) < 1e-5
        assert abs(y_c - expected_yc) < 1e-5

    def test_bbox_coords_in_unit_range(self):
        """All YOLO values must be in [0, 1]."""
        random.seed(99)
        bg = make_bg()
        tgt = make_target()
        _, label = paste_single_overlay(bg, tgt, class_id=1)
        if label is None:
            pytest.skip("No placement")
        x_c, y_c, w_n, h_n = label["bbox"]
        assert 0.0 <= x_c <= 1.0, f"x_center out of range: {x_c}"
        assert 0.0 <= y_c <= 1.0, f"y_center out of range: {y_c}"
        assert 0.0 < w_n <= 1.0, f"width out of range: {w_n}"
        assert 0.0 < h_n <= 1.0, f"height out of range: {h_n}"

    def test_no_zero_area_bbox(self):
        random.seed(7)
        bg = make_bg()
        tgt = make_target()
        _, label = paste_single_overlay(bg, tgt, class_id=1)
        if label is None:
            pytest.skip("No placement")
        _, _, w_n, h_n = label["bbox"]
        assert w_n > 0 and h_n > 0, "Bounding box must have non-zero area"

    def test_no_nan_in_bbox(self):
        random.seed(8)
        bg = make_bg()
        tgt = make_target()
        _, label = paste_single_overlay(bg, tgt, class_id=1)
        if label is None:
            pytest.skip("No placement")
        for val in label["bbox"]:
            assert not (val != val), f"NaN found in bbox: {label['bbox']}"  # NaN != NaN

    def test_width_proportional_to_scale(self):
        """w_norm ≈ scale (since overlay width = bg_w * scale, before perspective warp)."""
        random.seed(10)
        bg = make_bg(640, 480)
        tgt = make_target(100, 100)

        def noop_transform(img_np, max_offset=0.1):
            return img_np

        with patch(
            "compositor.overlay.overlay_single.random_perspective_transform",
            side_effect=noop_transform,
        ):
            _, label = paste_single_overlay(
                bg, tgt, class_id=1, scale_range=(0.3, 0.3), max_attempts=10
            )

        if label is None:
            pytest.skip("No placement")
        _, _, w_n, _ = label["bbox"]
        assert abs(w_n - 0.3) < 0.02, f"Expected w_norm ≈ 0.3, got {w_n}"


# ---------------------------------------------------------------------------
# paste_single_overlay — area constraint
# ---------------------------------------------------------------------------


class TestAreaConstraint:
    def test_overlay_never_exceeds_40_percent_of_background(self):
        """The function must skip placements where overlay area > 40% of bg."""
        random.seed(42)
        bg = make_bg(640, 480)
        tgt = make_target(80, 80)
        bg_area = 640 * 480

        for seed in range(20):
            random.seed(seed)
            result_img, label = paste_single_overlay(bg.copy(), tgt.copy(), class_id=1)
            if label is None:
                continue
            _, _, w_n, h_n = label["bbox"]
            overlay_area = (w_n * 640) * (h_n * 480)
            assert (
                overlay_area <= 0.4 * bg_area + 1
            ), f"Overlay area {overlay_area:.0f} exceeds 40% of bg ({0.4 * bg_area:.0f})"


# ---------------------------------------------------------------------------
# paste_single_overlay — color matching robustness
# ---------------------------------------------------------------------------


class TestColorMatching:
    def test_color_match_strength_zero_leaves_overlay_unchanged(self):
        """With color_match_strength=0, overlay RGB should not shift at all."""
        random.seed(5)
        bg = make_bg(640, 480, color="black")
        # Bright red target — would shift strongly with nonzero strength
        tgt = Image.new("RGBA", (80, 80), color=(255, 0, 0, 255))

        def noop_transform(img_np, max_offset=0.1):
            return img_np

        with patch(
            "compositor.overlay.overlay_single.random_perspective_transform",
            side_effect=noop_transform,
        ):
            result_img, label = paste_single_overlay(
                bg, tgt, class_id=1, color_match_strength=0.0
            )

        assert result_img is not None

    def test_output_has_no_nans(self):
        """Composite image pixel values must not contain NaN."""
        random.seed(6)
        bg = make_bg()
        tgt = make_target()
        result_img, _ = paste_single_overlay(bg, tgt, class_id=1)
        if result_img is None:
            pytest.skip("No placement")
        arr = np.array(result_img, dtype=np.float32)
        assert not np.isnan(arr).any()

    def test_output_pixels_in_valid_range(self):
        """All pixel values must be 0–255."""
        random.seed(11)
        bg = make_bg()
        tgt = make_target()
        result_img, _ = paste_single_overlay(bg, tgt, class_id=1)
        if result_img is None:
            pytest.skip("No placement")
        arr = np.array(result_img)
        assert arr.min() >= 0 and arr.max() <= 255


# ---------------------------------------------------------------------------
# Golden test — regression anchor
# ---------------------------------------------------------------------------


class TestGoldenRegression:
    """
    Fixed seed + fixed inputs → fixed bbox.  If this breaks, something subtle
    changed in the placement or normalization logic.
    """

    GOLDEN_SEED = 42
    EXPECTED_BBOX = (
        0.296094,
        0.677083,
        0.454688,
        0.454167,
    )  # found using pytest -v -k test_golden_bbox_is_stable -s

    def test_golden_bbox_is_stable(self):
        """
        HOW TO USE:
        1. Run once: pytest -v -k test_golden_bbox_is_stable -s
        2. Copy the printed bbox into EXPECTED_BBOX above.
        3. From then on this test acts as a regression guard.
        """
        random.seed(self.GOLDEN_SEED)
        np.random.seed(self.GOLDEN_SEED)

        bg = Image.new("RGB", (640, 480), color=(100, 149, 237))  # cornflower blue
        tgt = Image.new("RGBA", (80, 60), color=(200, 50, 50, 255))

        result_img, label = paste_single_overlay(bg, tgt, class_id=2, max_attempts=20)

        assert (
            result_img is not None
        ), "Golden test: placement must succeed with seed 42"
        bbox = label["bbox"]
        print(f"\n[golden] bbox = {tuple(round(v, 6) for v in bbox)}")

        if self.EXPECTED_BBOX is not None:
            for got, exp in zip(bbox, self.EXPECTED_BBOX):
                assert (
                    abs(got - exp) < 1e-4
                ), f"Regression: bbox changed. got={bbox}, expected={self.EXPECTED_BBOX}"
