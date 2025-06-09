import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def apply_global_augmentations(
    image: Image.Image,
    brightness_range=(0.5, 1.2),
    contrast_range=(0.5, 1.6),
    blur_prob=0.9,
    blur_kernel_range=(41, 51),
    jpeg_compression_prob=0.5,
    jpeg_quality_range=(50, 95)
):
    """
    Applies global augmentations to the whole composite image.
    Augmentations include brightness, contrast, blur, and JPEG compression.
    """
    img = image.copy()

    # Brightness
    if brightness_range:
        brightness_factor = random.uniform(*brightness_range)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Contrast
    if contrast_range:
        contrast_factor = random.uniform(*contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Convert to NumPy for OpenCV augmentations
    img_np = np.array(img)

    # Blur
    if random.random() < blur_prob:
        k = random.choice(range(blur_kernel_range[0], blur_kernel_range[1] + 1, 2))  # must be odd
        img_np = cv2.GaussianBlur(img_np, (k, k), 0)

    # JPEG compression (simulate lossy encoding)
    if random.random() < jpeg_compression_prob:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(*jpeg_quality_range)]
        _, enc_img = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)

    # Convert back to PIL
    return Image.fromarray(img_np)

if __name__=='__main__':
    img = Image.open('/Users/kvaikunthan/Desktop/data_generation/test_data/Task01_Sawfish.png').convert("RGB")

    cv2.imshow('hi', cv2.cvtColor(np.array(apply_global_augmentations(img)), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows