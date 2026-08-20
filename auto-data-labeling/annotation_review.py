import gradio as gr
import json
from pathlib import Path
from PIL import Image, ImageDraw
from utils import (
    draw_JSON_kpts,
    export_to_yolo, 
)

# Fetch JSON predictions
json_path = Path("./Test_JSON/raw_predictions.json")
image_dir = Path("./Test_Images")

with open(json_path, "r") as f:
    raw_preds = json.load(f)

img_fn = list(raw_preds.keys())

with gr.Blocks(title="Underwater Gate Pose Annotator") as app:

    # State variable to keep track of index
    curr_idx = gr.State(0)

    # Image display
    with gr.Row():
        img_display = gr.Image(type="pil", interactive=True, label="Current Gate Frame")

    # Corner selections for adjustment
    with gr.Row():
        corner_selector = gr.Radio(
            choices=["TL", "TR", "BL", "BR"], 
            value="TL", 
            label="Corner to Adjust", 
            interactive=True
        )

    # Accept button
    with gr.Row():
        btn_accept = gr.Button("Accept & Export")

    # ---------------------------------------
    #           HELPER FUNCTIONS
    # ---------------------------------------

    # Display each set of predictions on its corresponding image
    def render_image(index):
        if index >= len(img_fn):
            return Image.open("alldone.png").convert("RGB")
        filename = img_fn[index]
        img_path = image_dir / filename
        kpts = raw_preds[filename]
        return draw_JSON_kpts(img_path, kpts)

    # Logic for updating keypoints
    def update_kp(index, corner, evt: gr.SelectData):
        x, y = evt.index

        filename = img_fn[index]

        raw_preds[filename][corner] = [x, y]

        return render_image(index)

    # Accepting the corner points
    def accept_and_next(index):
        if index >= len(img_fn):
            return index, render_image(index)
        
        filename = img_fn[index]
        points = raw_preds[filename]

        img_path = image_dir / filename
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        export_to_yolo(filename, points, img_width, img_height)

        # Move onto next image
        new_index = index + 1
        new_img = render_image(new_index)

        return new_index, new_img

    # ---------------------------------------
    #            BUTTON CONFIGS
    # ---------------------------------------

    app.load(fn=render_image, inputs=curr_idx, outputs=img_display)

    img_display.select(
        fn=update_kp,
        inputs=[curr_idx, corner_selector],
        outputs=img_display,
    )

    btn_accept.click(
        fn=accept_and_next,
        inputs=curr_idx,
        outputs=[curr_idx, img_display]
    )

if __name__ == "__main__":
    app.launch()