import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor
import numpy as np
import matplotlib

image_path = "Gate1.png"
prompts_to_try = ["vertical black pole"] # "horizontal black pole"
confidence_threshold = 0.5
post_ratio_thresh = 0.5

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=3)
    return image

def draw_global_corners(image, combined_boxes):
    corners = get_labeled_corners(combined_boxes, image.width)
    if not corners:
        return image

    draw = ImageDraw.Draw(image)
    r = 12  # Radius of the keypoint dot

    try:
        large_font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        large_font = ImageFont.load_default(size=30)

    for label_name, (cx, cy) in corners.items():
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="cyan", outline="white", width=2)
        draw.text((cx + r + 5, cy - 8), label_name, fill="yellow", font=large_font)

    return image

def keep_largest_component(mask_tensor, original_box):
    """
    Helper function for segmentations that identify discontinuous bodies as one object.
    """
    # Convert tensor into a cv2 compatible mask
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np)

    # Ignore if there is no separation
    if num_labels <= 1:
        return mask_np, original_box

    # Isolate pixle counts for each detected label
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    # Only keep the largest label
    clean_mask_np = (labels == largest_label).astype(np.uint8)
    clean_mask_tensor = torch.from_numpy(clean_mask_np).to(mask_tensor.device)

    x_min = stats[largest_label, cv2.CC_STAT_LEFT]
    y_min = stats[largest_label, cv2.CC_STAT_TOP]
    width = stats[largest_label, cv2.CC_STAT_WIDTH]
    height = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Draw bounding box
    clean_box = torch.tensor([x_min, y_min, x_min + width, y_min + height], device=mask_tensor.device)

    return clean_mask_tensor, clean_box

def get_labeled_corners(combined_boxes, image_width):
    """
    Calculates and returns named keypoint coordinates for dataset labeling
    """
    if len(combined_boxes) == 0:
        return {}

    if len(combined_boxes) == 1:
        x_min, y_min, x_max, y_max = combined_boxes[0].tolist()
        x_center = (x_min + x_max) / 2
        
        # Compare post center to image center
        if x_center < (image_width / 2):
            return {
                "TL": (x_center, y_min),
                "BL": (x_center, y_max)
            }
        else:
            return {
                "TR": (x_center, y_min),
                "BR": (x_center, y_max)
            }

    # Sort boxes left-to-right by x_min
    sorted_ind = torch.argsort(combined_boxes[:, 0])
    sorted_boxes = combined_boxes[sorted_ind]

    lx_min, ly_min, lx_max, ly_max = sorted_boxes[0].tolist()
    rx_min, ry_min, rx_max, ry_max = sorted_boxes[1].tolist()

    left_x_center = (lx_min + lx_max) / 2
    right_x_center = (rx_min + rx_max) / 2

    return {
        "TL": (left_x_center, ly_min),
        "TR": (right_x_center, ry_min),
        "BL": (left_x_center, ly_max),
        "BR": (right_x_center, ry_max),
    }

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
 
print("Loading model weights into memory...")
load_start = time.time()
 
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
 
load_end = time.time()
print(f"Model loaded in: {load_end - load_start:.2f} seconds\n")
 
image = Image.open(image_path).convert("RGB")
 
print(f"Running sequential inference for: {prompts_to_try}...")
inference_start = time.time()

# Setup empty lists to accumulate results across all prompts
all_masks = []
all_boxes = []
all_scores = []

# Loop through each prompt
for prompt_text in prompts_to_try:
    print(f"  -> Processing: '{prompt_text}'")
    inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(device)
     
    with torch.no_grad():
        outputs = model(**inputs)
     
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=confidence_threshold,
        mask_threshold=confidence_threshold,
        target_sizes=[image.size[::-1]], 
    )[0]
    
    # Only append if the model actually found a matching mask for this specific prompt
    if len(results["masks"]) > 0:
        masks = results["masks"]
        boxes = results["boxes"]
        scores = results["scores"]
        
        areas = []
        # Convert tensor masks to numpy arrays for OpenCV
        masks_np = masks.cpu().numpy().astype(np.uint8)
        
        for m in masks_np:
            # Scale 0/1 binary mask to 0/255 for cv2
            m_255 = m * 255
            contours, _ = cv2.findContours(m_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour area within this specific mask
                max_area = max([cv2.contourArea(c) for c in contours])
                areas.append(max_area)
            else:
                areas.append(0.0)
                
        # Sort indices by area in descending order
        sorted_indices = np.argsort(areas)[::-1].tolist()

        valid_indices = []
        if len(sorted_indices) > 0:

            # Always keep the largest area (the primary post)
            largest_idx = sorted_indices[0]
            largest_area = areas[largest_idx]
            
            if largest_area > 0:
                valid_indices.append(largest_idx)
            
            # Evaluate the second largest if it exists
            if len(sorted_indices) > 1:
                second_idx = sorted_indices[1]
                second_area = areas[second_idx]
                
                # Check if the second post meets the 0.7 ratio threshold
                if (second_area / largest_area) >= post_ratio_thresh:
                    valid_indices.append(second_idx)
                else:
                    print(f"    -> Discarding second post. Ratio: {(second_area/largest_area):.2f} < {post_ratio_thresh}")
        
        # Filter the tensors
        filtered_masks = masks[valid_indices]
        filtered_boxes = boxes[valid_indices]
        filtered_scores = scores[valid_indices]

        cleaned_masks = []
        cleaned_boxes = []
        
        for mask, box in zip(filtered_masks, filtered_boxes):
            c_mask, c_box = keep_largest_component(mask, box)
            cleaned_masks.append(c_mask)
            cleaned_boxes.append(c_box)
            
        filtered_masks = torch.stack(cleaned_masks)
        filtered_boxes = torch.stack(cleaned_boxes)
        
        all_masks.append(filtered_masks)
        all_boxes.append(filtered_boxes)
        all_scores.append(filtered_scores)
 
inference_end = time.time()
elapsed_time = inference_end - inference_start
 
print("\nInference completed successfully!")
print(f"Total sequential execution time: {elapsed_time:.4f} seconds")

# Combine all collected data and draw
if all_masks:
    combined_masks = torch.cat(all_masks, dim=0)
    combined_boxes = torch.cat(all_boxes, dim=0)
    combined_scores = torch.cat(all_scores, dim=0)
    
    print(f"Found {len(combined_masks)} instance(s) combined (Top 2 per prompt)")
    print(f"Scores: {combined_scores.tolist()}")

    result = overlay_masks(image, combined_masks)
    result = draw_boxes(result, combined_boxes)
    result = draw_global_corners(result, combined_boxes)

    result.save("output.png")
    print("Saved combined visualization to 'output.png'")
else:
    print("No instances found for any of the provided prompts.")