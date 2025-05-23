# preprocess_png_to_concatenated_tiles.py

import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError  # Using PIL to read and process PNG
from tqdm import tqdm
import gc

Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
class PreprocessConfig:
    data_dir = "/home/gpl_homee/CVDL_Final/data"  # Directory containing train.csv
    input_png_dir = "/home/gpl_homee/CVDL_Final/data/train_png"  # Directory containing source PNGs
    output_dir = "/home/gpl_homee/CVDL_Final/data/train_concatenated_pngs_from_png_source"  # Output directory for concatenated PNGs

    tile_size = 128  # Size of each tile
    grid_side_count = 8  # Grid dimension (e.g., 8x8)
    num_tiles_total = grid_side_count ** 2  # Total number of tiles to extract

    foreground_threshold = 230  # Threshold for foreground detection

# --- Tile extraction logic ---
def get_tiles_from_loaded_image(image_np, tile_size, num_tiles_to_extract, fg_threshold=230):
    """
    Extract foreground tiles from a given NumPy image.
    image_np: Loaded PNG image as NumPy array (HxWxC, RGB)
    """
    tiles_list = []
    region_h, region_w, _ = image_np.shape
    tiles_in_row = region_w // tile_size
    tiles_in_col = region_h // tile_size

    if tiles_in_row == 0 or tiles_in_col == 0:
        # Source image is too small to extract any full tile
        print(f"Warning: Source PNG ({region_w}x{region_h}) is too small to extract {tile_size}x{tile_size} tiles.")
        for _ in range(num_tiles_to_extract):
            white_tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            tiles_list.append(white_tile)
        return np.array(tiles_list)

    candidate_tiles = []
    for i in range(tiles_in_col):
        for j in range(tiles_in_row):
            y_start, y_end = i * tile_size, (i + 1) * tile_size
            x_start, x_end = j * tile_size, (j + 1) * tile_size
            tile = image_np[y_start:y_end, x_start:x_end, :]
            
            if np.mean(tile) < fg_threshold:
                candidate_tiles.append(tile)
            elif not candidate_tiles and (i * tiles_in_row + j) >= (tiles_in_col * tiles_in_row - num_tiles_to_extract):
                candidate_tiles.append(tile)

    if not candidate_tiles:
        all_possible_tiles = []
        for i in range(tiles_in_col):
            for j in range(tiles_in_row):
                y_start, y_end = i * tile_size, (i + 1) * tile_size
                x_start, x_end = j * tile_size, (j + 1) * tile_size
                all_possible_tiles.append(image_np[y_start:y_end, x_start:x_end, :])
        if all_possible_tiles:
            candidate_tiles = all_possible_tiles[:num_tiles_to_extract] if len(all_possible_tiles) >= num_tiles_to_extract else all_possible_tiles

    if len(candidate_tiles) > num_tiles_to_extract:
        tiles_list = candidate_tiles[:num_tiles_to_extract]
    else:
        tiles_list = candidate_tiles

    while len(tiles_list) < num_tiles_to_extract:
        white_tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        tiles_list.append(white_tile)
        if not tiles_list and len(tiles_list) >= num_tiles_to_extract:
            break
    return np.array(tiles_list) if tiles_list else np.array([np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)] * num_tiles_to_extract)

# --- Tile concatenation logic ---
def tile_concat_custom(tiles_np_array, tile_size, grid_side_count):
    num_tiles_total = grid_side_count ** 2
    if tiles_np_array.shape[0] != num_tiles_total:
        print(f"Warning: Number of tiles ({tiles_np_array.shape[0]}) does not match expected total ({num_tiles_total}). Will use available tiles or pad.")
        if tiles_np_array.shape[0] < num_tiles_total:
            num_to_pad = num_tiles_total - tiles_np_array.shape[0]
            white_tile_data = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            padding = np.array([white_tile_data] * num_to_pad)
            if tiles_np_array.shape[0] == 0:
                tiles_np_array = padding
            else:
                tiles_np_array = np.concatenate([tiles_np_array, padding], axis=0)

    channels = tiles_np_array.shape[-1]
    concatenated_image = np.zeros(
        (tile_size * grid_side_count, tile_size * grid_side_count, channels),
        dtype=np.uint8
    )
    for i in range(grid_side_count):
        for j in range(grid_side_count):
            idx = i * grid_side_count + j
            if idx < tiles_np_array.shape[0]:
                concatenated_image[
                    i * tile_size:(i + 1) * tile_size,
                    j * tile_size:(j + 1) * tile_size, :
                ] = tiles_np_array[idx]
            else:
                concatenated_image[
                    i * tile_size:(i + 1) * tile_size,
                    j * tile_size:(j + 1) * tile_size, :
                ] = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    return concatenated_image

# --- Main processing function ---
def process_pngs_to_concatenated_tiles(config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        print(f"Created output directory: {config.output_dir}")

    if not os.path.exists(config.input_png_dir):
        print(f"Error: Input PNG directory '{config.input_png_dir}' not found.")
        return

    train_csv_path = os.path.join(config.data_dir, "train.csv")
    if not os.path.exists(train_csv_path):
        print(f"Error: train.csv (or your specified CSV) not found at {train_csv_path}.")
        return

    df = pd.read_csv(train_csv_path)

    print(f"Processing {len(df)} images listed in {train_csv_path} from source PNGs in {config.input_png_dir}.")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing source PNGs"):
        image_id = row.image_id
        source_png_path = os.path.join(config.input_png_dir, f"{image_id}.png")

        if not os.path.exists(source_png_path):
            print(f"Warning: Source PNG file {source_png_path} not found, skipping.")
            continue

        try:
            source_pil_image = Image.open(source_png_path).convert('RGB')
            source_np_image = np.array(source_pil_image)

            tiles_np_array = get_tiles_from_loaded_image(
                source_np_image,
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

            del source_pil_image, source_np_image, tiles_np_array, concatenated_image_np, concatenated_img_pil
            if (index + 1) % 100 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error occurred while processing source PNG {source_png_path}: {e}")
            import traceback
            traceback.print_exc()

    print("All source PNGs processed and tiles concatenated!")

# --- Run the script ---
if __name__ == "__main__":
    config = PreprocessConfig()

    config.input_png_dir = "/home/gpl_homee/CVDL_Final/data/train_png"
    config.output_dir = "/home/gpl_homee/CVDL_Final/data/train_concatenated_64*256"
    config.grid_side_count = 8
    config.tile_size = 256
    config.num_tiles_total = config.grid_side_count ** 2

    print(f"Reading source PNGs from: {config.input_png_dir}")
    print(f"Each source PNG will be cut into {config.num_tiles_total} ({config.grid_side_count}x{config.grid_side_count}) tiles of size {config.tile_size}x{config.tile_size},")
    print(f"then concatenated into one image and saved to: {config.output_dir}")

    process_pngs_to_concatenated_tiles(config)
