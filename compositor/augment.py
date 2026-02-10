import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def augment(
    image: Image.Image,
    brightness_range: tuple = (0.5, 1.2),
    contrast_range: tuple = (0.5, 1.6),
    blur_prob: float = 0.3,  # much lower probability
    blur_kernel_range: tuple = (3, 5),  # very slight blur
    jpeg_compression_prob: float = 0.0,  # skip compression
) -> Image.Image:
    """
    Apply random augmentations to an input image.
    """
    img_np = np.array(image)

    img = Image.fromarray(img_np)

    # Brightness adjustment
    brightness_factor = random.uniform(*brightness_range)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Contrast adjustment
    contrast_factor = random.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    img_np = np.array(img)

    # Very slight Gaussian Blur
    if random.random() < blur_prob:
        k = random.choice(range(blur_kernel_range[0], blur_kernel_range[1] + 1, 2))  # must be odd
        img_np = cv2.GaussianBlur(img_np, (k, k), 0)

    return Image.fromarray(img_np)
