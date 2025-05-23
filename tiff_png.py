import os
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm

input_dir = "/home/gpl_homee/CVDL_Final/data/train_images"
output_dir = "/home/gpl_homee/CVDL_Final/data/train_png"

os.makedirs(output_dir, exist_ok=True)

def convert_wsi_to_png(image_id, level=1):
    path = os.path.join(input_dir, f"{image_id}.tiff")
    if not os.path.exists(path):
        print(f"[SKIP] {image_id}.tiff not found.")
        return
    slide = openslide.OpenSlide(path)
    if level >= slide.level_count:
        level = slide.level_count - 1
    dims = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, dims)
    region_np = np.array(region)[..., :3]  # remove alpha if exists
    Image.fromarray(region_np).save(os.path.join(output_dir, f"{image_id}.png"))
    slide.close()

# Example: convert all .tiff images in the folder
for fname in tqdm(os.listdir(input_dir)):
    if fname.endswith(".tiff"):
        image_id = fname.replace(".tiff", "")
        convert_wsi_to_png(image_id, level=1)
