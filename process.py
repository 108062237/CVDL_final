# preprocess_tiles_concatenated.py (New filename to distinguish)

import os
import pandas as pd
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm
import gc

# --- Configuration ---
class PreprocessConfig:
    data_dir = "/home/gpl_homee/CVDL_Final/data"
    output_dir = "/home/gpl_homee/CVDL_Final/data/train_concatenated_pngs"  # Directory to save concatenated images

    wsi_level_to_read = 1

    # Tile configuration
    tile_size = 128
    grid_side_count = 8  # If we want final image to be grid_side x grid_side tiles
    num_tiles_total = grid_side_count ** 2  # Total number of tiles to extract

    foreground_threshold = 230

# --- Tile extraction logic ---
def get_tiles_from_region(region_np, tile_size, num_tiles_to_extract, fg_threshold=230):
    tiles_list = []
    region_h, region_w, _ = region_np.shape
    tiles_in_row = region_w // tile_size
    tiles_in_col = region_h // tile_size

    if tiles_in_row == 0 or tiles_in_col == 0:
        for _ in range(num_tiles_to_extract):
            white_tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            tiles_list.append(white_tile)
        return np.array(tiles_list)

    candidate_tiles = []
    for i in range(tiles_in_col):
        for j in range(tiles_in_row):
            y_start, y_end = i * tile_size, (i + 1) * tile_size
            x_start, x_end = j * tile_size, (j + 1) * tile_size
            tile = region_np[y_start:y_end, x_start:x_end, :]
            if np.mean(tile) < fg_threshold:
                candidate_tiles.append(tile)
            elif not candidate_tiles and (i*tiles_in_row + j) >= (tiles_in_col*tiles_in_row - num_tiles_to_extract):
                candidate_tiles.append(tile)

    if not candidate_tiles:
        all_possible_tiles = []
        for i in range(tiles_in_col):
            for j in range(tiles_in_row):
                y_start, y_end = i * tile_size, (i + 1) * tile_size
                x_start, x_end = j * tile_size, (j + 1) * tile_size
                all_possible_tiles.append(region_np[y_start:y_end, x_start:x_end, :])
        if all_possible_tiles:
            candidate_tiles = all_possible_tiles[:num_tiles_to_extract] if len(all_possible_tiles) >= num_tiles_to_extract else all_possible_tiles

    if len(candidate_tiles) > num_tiles_to_extract:
        tiles_list = candidate_tiles[:num_tiles_to_extract]
    else:
        tiles_list = candidate_tiles

    while len(tiles_list) < num_tiles_to_extract:
        white_tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        tiles_list.append(white_tile)
        if not tiles_list and len(tiles_list) >= num_tiles_to_extract :
            break
    return np.array(tiles_list) if tiles_list else np.array([np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)] * num_tiles_to_extract)

# --- Tile concatenation logic ---
def tile_concat_custom(tiles_np_array, tile_size, grid_side_count):
    num_tiles_total = grid_side_count ** 2
    if tiles_np_array.shape[0] != num_tiles_total:
        raise ValueError(f"Number of tiles ({tiles_np_array.shape[0]}) does not match expected total ({num_tiles_total}).")

    channels = tiles_np_array.shape[-1]
    concatenated_image = np.zeros(
        (tile_size * grid_side_count, tile_size * grid_side_count, channels),
        dtype=np.uint8
    )

    for i in range(grid_side_count):
        for j in range(grid_side_count):
            idx = i * grid_side_count + j
            concatenated_image[
                i * tile_size:(i + 1) * tile_size,
                j * tile_size:(j + 1) * tile_size, :
            ] = tiles_np_array[idx]
    return concatenated_image

# --- Main processing function ---
def process_all_wsi_and_concat(config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        print(f"Created output directory: {config.output_dir}")

    train_csv_path = os.path.join(config.data_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        print(f"Error: train.csv not found at {train_csv_path}.")
        return

    df = pd.read_csv(train_csv_path)
    df_to_process = df

    print(f"Total {len(df_to_process)} WSI to process.")

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Processing WSI and concatenating"):
        image_id = row.image_id
        wsi_path = os.path.join(config.data_dir, "train_images", f"{image_id}.tiff")

        if not os.path.exists(wsi_path):
            print(f"Warning: Image file {wsi_path} not found, skipping.")
            continue

        try:
            slide = openslide.OpenSlide(wsi_path)
            level_to_read_actual = config.wsi_level_to_read
            if config.wsi_level_to_read >= slide.level_count:
                print(f"Warning: Image {image_id} level {config.wsi_level_to_read} does not exist. Using highest available level {slide.level_count - 1}.")
                level_to_read_actual = slide.level_count - 1
            
            region_dims = slide.level_dimensions[level_to_read_actual]
            region_pil = slide.read_region(location=(0, 0), level=level_to_read_actual, size=region_dims)
            region_np = np.array(region_pil)
            
            if region_np.shape[-1] == 4:
                region_np = region_np[..., :3]
            slide.close()

            tiles_np_array = get_tiles_from_region(
                region_np,
                config.tile_size,
                config.num_tiles_total,
                config.foreground_threshold
            )

            concatenated_image_np = tile_concat_custom(
                tiles_np_array,
                config.tile_size,
                config.grid_side_count
            )

            concatenated_img_pil = Image.fromarray(concatenated_image_np.astype(np.uint8))
            output_filename = os.path.join(config.output_dir, f"{image_id}.png")
            concatenated_img_pil.save(output_filename)
            
            del region_np, tiles_np_array, region_pil, concatenated_image_np, concatenated_img_pil
            if (index + 1) % 50 == 0:
                 gc.collect()

        except openslide.OpenSlideError as e:
            print(f"OpenSlide error occurred while processing image {image_id}: {e}, skipping.")
        except Exception as e:
            print(f"Unknown error occurred while processing image {image_id}: {e}, skipping.")
            import traceback
            traceback.print_exc()

    print("All images processed and concatenated!")

# --- Run preprocessing ---
if __name__ == "__main__":
    config = PreprocessConfig()
    
    # Example: if you want to get 8x8=64 tiles of size 128x128
    config.grid_side_count = 8
    config.tile_size = 128
    config.num_tiles_total = config.grid_side_count ** 2
    config.wsi_level_to_read = 1

    print(f"Reading from WSI level {config.wsi_level_to_read}.")
    print(f"Each WSI will extract {config.num_tiles_total} ({config.grid_side_count}x{config.grid_side_count}) tiles of size {config.tile_size}x{config.tile_size},")
    print(f"then concatenate into one image and save to: {config.output_dir}")
    
    process_all_wsi_and_concat(config)
