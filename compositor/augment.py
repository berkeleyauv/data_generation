import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def random_hue_saturation(img_np, max_hue_shift=15, sat_scale_range=(0.8, 1.2)):
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue_shift = random.uniform(-max_hue_shift, max_hue_shift)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

    sat_scale = random.uniform(*sat_scale_range)
    hsv[..., 1] *= sat_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)

    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def augment(
    image: Image.Image,
    hue_shift_max=15,
    sat_scale_range=(0.8, 1.2),
    brightness_range=(0.5, 1.2),
    contrast_range=(0.5, 1.6),
    blur_prob=0.9,
    blur_kernel_range=(41, 91),
    jpeg_compression_prob=0.6,
    jpeg_quality_range=(50, 95)
):
    img_np = np.array(image)

    # Hue & Saturation (color jitter)
    img_np = random_hue_saturation(img_np, hue_shift_max, sat_scale_range)

    img = Image.fromarray(img_np)

    # Brightness
    brightness_factor = random.uniform(*brightness_range)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Contrast
    contrast_factor = random.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    img_np = np.array(img)

    # Blur
    if random.random() < blur_prob:
        k = random.choice(range(blur_kernel_range[0], blur_kernel_range[1] + 1, 2))
        img_np = cv2.GaussianBlur(img_np, (k, k), 0)

    # JPEG compression simulation
    if random.random() < jpeg_compression_prob:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(*jpeg_quality_range)]
        _, enc_img = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)

    return Image.fromarray(img_np)


if __name__=='__main__':
    img = Image.open('/Users/kvaikunthan/Desktop/data_generation/test_data/Task01_Sawfish.png').convert("RGB")

    cv2.imshow('hi', cv2.cvtColor(np.array(augment(img)), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows