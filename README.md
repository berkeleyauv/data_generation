#Data Generation
Scripts for generating synthetic datasets for deep learning. Includes random augmentation and supports dynamic overlays, automatic class mapping, and even distribution of target images across all backgrounds.
** Step by Step demo at the bottom.


## Directory Setup
This code expects:
```bash
data_generation/        #repo root/project root
├── backgrounds/        # Folder containing background images (.jpg/.png)
├── targets_original/   # Folder containing real target images (.jpg/.png)
├── targets_fake/       # Folder containing fake target images (.jpg/.png)
```
backgrounds - In this you would put all the backgrounds you want to generate image target overlays over.


targets_original - Put all the cropped target images you want to lay on backgrounds here.


targets_fake - Put all the fake target images here.


## Usage
Run from the project root:


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


to have multiple target images placed on your image, instead use generate_multiple.py.


```bash
python generate_multiple.py \
 --backgrounds_dir backgrounds \
 --real_targets_dir targets_original \
 --output_img_dir output_images \
 --output_yolo_dir output_yolo \
 --max_attempts 20 \
 --num_backgrounds 10
```
with fake images (adds a single fake image (randomly chosen from a set of fake images) along with real target images)


```bash
python generate.py \
 --backgrounds_dir backgrounds \
 --real_targets_dir targets_original \
 --fake_targets_dir targets_fake \
 --output_img_dir output_images \
 --output_yolo_dir output_yolo \
 --max_attempts 20 \
 --num_backgrounds 10
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


# Step by Step Demo:
What is the setup? How do you handle backgrounds and targets? 


**Case:**  You are trying to generate data to train a YOLOv8 model to recognize and move towards a beachball in a pool spotted in your camera.


## 1. Setup 
If you haven't already,
Clone the repo: 
git clone https://github.com/berkeleyauv/data_generation.git


Find your backgrounds! Since your situation was in a pool, these backgrounds are also in a pool or underwater so the model that is trained on your data has the easiest time detecting beachballs specifically in pools and not completely random images.


Collect your targets! Make sure target images vary in orientation, lighting, angles, and scale to avoid overfitting. For ~10,000 images, 50-200 target images is a good amount.


Collect your fakes! Might not be needed, but if you do choose to use it, find photos of beachball like items, *items that the model could confuse for beachballs* and crop them using the same technique you used to create your target images. The ideal amount if you follow the recommended target image amount above might be ~ 10-20 fake images.


## 2. Call generate.py/generate_multiple.py 
If you’ve cloned the repo properly you should have two files, generate.py and generate_multiple.py


In the case of detecting the beachball, generate.py will place one singular beachball image on a background image while generate_multiple.py will place several ~10 beachballs. It all depends on how many beachballs you are planning to detect at one. Most cases, it is best to just use generate.py as it likely be more accurate than using generate_multiple.py.


## 3. Cropping your Images [How we did it]
As for how you place the images, 
this repo’s purpose of detecting objects on fake backgrounds you need to have some cropped overlays of the actual beachballs. 

One way to do so if utilize [roboflow](https://roboflow.com/) and it's labelling tool.  

1. Input your dataset of target images into a Roboflow project. 
2. Manually Label some a set of **10-20** images, let Roboflow auto-label the rest but make sure to review the beachball is being boxed properly. 
3. Look at your projects versions, export a version using Roboflows cropping images tool. 
4. Download this dataset and replace your current target images folder with this updated set. 

You now have cropped target images!
Repeat this process for your fake targets if needed. 

## 4. Generating your data. 
The next step is to actually generate your data. 

Make sure your data is in the following directories or adjust the command to take in your directory paths if they are named differently. 

target_original : cropped target images
target_fake : cropped fake images
backgrounds : your backgrounds

Now that you have the set up, 

Run the following: 

```
python generate.py \
  --backgrounds_dir backgrounds \
  --real_targets_dir targets_original \
  --output_img_dir output_images \
  --output_yolo_dir output_yolo \
  --max_attempts 20 \
  --num_backgrounds 15000
```

This will generate 15000 (will only do 15000 if you have 15000+ backgrounds) background images with your target image overlays. 

If you want to generate mulitple overlays just replace ```generate.py``` with ```generate_multiple.py``` and if you want to change the number of images produce just change the value that comes after ```--num_backgrounds```. 

## 5. Extra: Visualizing your labels

If you want to generate labels for your images just run ```python3 visualize_labels.py``` in your terminal no arguments needed. 

