import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance


def random_hue_saturation(
    img_np: np.ndarray,
    max_hue_shift: int = 15,
    sat_scale_range: tuple = (0.8, 1.2),
) -> np.ndarray:
    """
    Randomly adjust hue and saturation of an image.

    Args:
        img_np: Input image as a NumPy RGB array.
        max_hue_shift: Maximum hue shift (in OpenCV HSV hue scale, 0–180).
        sat_scale_range: Range for scaling saturation.

    Returns:
        Augmented RGB image as a NumPy array.
    """
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Random hue shift
    hue_shift = random.uniform(-max_hue_shift, max_hue_shift)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

    # Random saturation scaling
    sat_scale = random.uniform(*sat_scale_range)
    hsv[..., 1] *= sat_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def augment(
    image: Image.Image,
    hue_shift_max: int = 15,
    sat_scale_range: tuple = (0.8, 1.2),
    brightness_range: tuple = (0.5, 1.2),
    contrast_range: tuple = (0.5, 1.6),
    blur_prob: float = 0.9,
    blur_kernel_range: tuple = (41, 91),
    jpeg_compression_prob: float = 0.6,
    jpeg_quality_range: tuple = (50, 95),
) -> Image.Image:
    """
    Apply random augmentations to an input image.

    Args:
        image: Input PIL image.
        hue_shift_max: Maximum hue shift in HSV space.
        sat_scale_range: Range for scaling saturation.
        brightness_range: Brightness scaling factor range.
        contrast_range: Contrast scaling factor range.
        blur_prob: Probability of applying Gaussian blur.
        blur_kernel_range: Range of odd kernel sizes for blur.
        jpeg_compression_prob: Probability of applying JPEG compression.
        jpeg_quality_range: JPEG quality range (1–100).

    Returns:
        Augmented PIL Image.
    """
    img_np = np.array(image)

    # Hue & Saturation adjustment
    img_np = random_hue_saturation(img_np, hue_shift_max, sat_scale_range)

    img = Image.fromarray(img_np)

    # Brightness adjustment
    brightness_factor = random.uniform(*brightness_range)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Contrast adjustment
    contrast_factor = random.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    img_np = np.array(img)

    # Apply Gaussian Blur
    if random.random() < blur_prob:
        k = random.choice(range(blur_kernel_range[0], blur_kernel_range[1] + 1, 2))  # odd kernel
        img_np = cv2.GaussianBlur(img_np, (k, k), 0)

    # Simulate JPEG Compression
    if random.random() < jpeg_compression_prob:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(*jpeg_quality_range)]
        _, enc_img = cv2.imencode(".jpg", img_np, encode_param)
        img_np = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img_np)


if __name__ == "__main__":
    # Example usage
    test_path = "/Users/kvaikunthan/Desktop/data_generation/test_data/Task01_Sawfish.png"
    img = Image.open(test_path).convert("RGB")

    augmented_img = augment(img)

    # Show with OpenCV
    cv2.imshow("Augmented", cv2.cvtColor(np.array(augmented_img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
