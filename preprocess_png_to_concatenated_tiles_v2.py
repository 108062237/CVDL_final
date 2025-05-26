# preprocess_png_to_concatenated_tiles.py

import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import gc

Image.MAX_IMAGE_PIXELS = None

# --- Configuration ---
class PreprocessConfig:
    data_dir = "/home/gpl_homee/CVDL_Final/data"  # Directory containing train.csv or clean_noise.csv
    # ❗建議：如果使用 clean_noise.csv，請修改下面的 train_csv_name
    train_csv_name = "clean_noise.csv" # 或者 "clean_noise.csv"
    input_png_dir = "/home/gpl_homee/CVDL_Final/data/train_png"  # Directory containing source PNGs
    output_dir = "/home/gpl_homee/CVDL_Final/data/train_concatenated_pngs_36*256_v2"  # Output directory for concatenated PNGs (建議用新目錄名區分)

    tile_size = 256  # Size of each tile (可以根據 if __name__ 中的設置調整)
    grid_side_count = 6  # Grid dimension (e.g., 8x8) (可以根據 if __name__ 中的設置調整)
    num_tiles_total = grid_side_count ** 2  # Total number of tiles to extract

    foreground_threshold = 230  # Threshold for foreground detection (越小代表組織越暗)

# --- Tile extraction logic (修改後) ---
def get_tiles_from_loaded_image(image_np, tile_size, num_tiles_to_extract, fg_threshold=230):
    """
    Extract top foreground tiles from a given NumPy image based on a quality score.
    image_np: Loaded PNG image as NumPy array (HxWxC, RGB)
    """
    region_h, region_w, channels = image_np.shape
    tiles_in_row = region_w // tile_size
    tiles_in_col = region_h // tile_size

    # 如果源圖像太小，無法提取任何完整的圖塊
    if tiles_in_row == 0 or tiles_in_col == 0:
        print(f"Warning: Source PNG ({region_w}x{region_h}) is too small to extract {tile_size}x{tile_size} tiles. Filling with white tiles.")
        white_tiles_list = [np.full((tile_size, tile_size, channels), 255, dtype=np.uint8) for _ in range(num_tiles_to_extract)]
        return np.array(white_tiles_list)

    scored_candidate_tiles = [] # 用於存儲 {'score': score, 'tile': tile_data}

    for i in range(tiles_in_col):
        for j in range(tiles_in_row):
            y_start, y_end = i * tile_size, (i + 1) * tile_size
            x_start, x_end = j * tile_size, (j + 1) * tile_size
            tile = image_np[y_start:y_end, x_start:x_end, :]
            
            mean_pixel_value = np.mean(tile)
            if mean_pixel_value < fg_threshold: # 越暗越可能是前景
                # 分數越高代表越可能是前景/組織 (fg_threshold - mean_pixel_value)
                # 也可以考慮其他評分方式，例如方差 np.var(tile)
                score = fg_threshold - mean_pixel_value
                scored_candidate_tiles.append({'score': score, 'tile': tile})

    # 按分數從高到低排序 (選擇最暗/最像組織的圖塊)
    scored_candidate_tiles.sort(key=lambda x: x['score'], reverse=True)

    selected_tiles_list = []
    # 從排序後的候選圖塊中選取
    for item in scored_candidate_tiles:
        if len(selected_tiles_list) < num_tiles_to_extract:
            selected_tiles_list.append(item['tile'])
        else:
            break # 已選夠數量

    # 如果選取的圖塊數量不足，用白色圖塊填充
    num_to_pad = num_tiles_to_extract - len(selected_tiles_list)
    if num_to_pad > 0:
        white_tile_data = np.full((tile_size, tile_size, channels), 255, dtype=np.uint8)
        for _ in range(num_to_pad):
            selected_tiles_list.append(white_tile_data)
    
    return np.array(selected_tiles_list)

# --- Tile concatenation logic (基本不變，稍作健壯性調整) ---
def tile_concat_custom(tiles_np_array, tile_size, grid_side_count):
    num_tiles_total = grid_side_count ** 2
    
    # 確保 tiles_np_array 至少有一個維度並且包含圖塊數據
    if tiles_np_array is None or tiles_np_array.ndim == 0 or tiles_np_array.shape[0] == 0:
        print(f"Warning: No tiles provided for concatenation. Filling with white image.")
        white_tile_data = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8) # 假設3通道
        tiles_np_array = np.array([white_tile_data] * num_tiles_total)

    # 如果提供的圖塊數量與預期不符，進行調整
    if tiles_np_array.shape[0] != num_tiles_total:
        print(f"Warning: Number of tiles ({tiles_np_array.shape[0]}) does not match expected total ({num_tiles_total}). Adjusting.")
        if tiles_np_array.shape[0] < num_tiles_total:
            num_to_pad = num_tiles_total - tiles_np_array.shape[0]
            # 確保 white_tile_data 的通道數與 tiles_np_array 一致
            channels_in_tiles = tiles_np_array.shape[-1] if tiles_np_array.ndim > 1 and tiles_np_array.shape[0] > 0 else 3
            white_tile_data = np.full((tile_size, tile_size, channels_in_tiles), 255, dtype=np.uint8)
            padding = np.array([white_tile_data] * num_to_pad)
            tiles_np_array = np.concatenate([tiles_np_array, padding], axis=0)
        else: # 圖塊過多，截取前面的
            tiles_np_array = tiles_np_array[:num_tiles_total]
            
    channels = tiles_np_array.shape[-1]
    concatenated_image = np.zeros(
        (tile_size * grid_side_count, tile_size * grid_side_count, channels),
        dtype=np.uint8 # 確保是 uint8
    )

    for i in range(grid_side_count):
        for j in range(grid_side_count):
            idx = i * grid_side_count + j
            # 這裡不需要再判斷 idx < tiles_np_array.shape[0]，因為前面已經保證了數量一致
            concatenated_image[
                i * tile_size:(i + 1) * tile_size,
                j * tile_size:(j + 1) * tile_size, :
            ] = tiles_np_array[idx]
            
    return concatenated_image

# --- Main processing function (修改了CSV文件名讀取方式) ---
def process_pngs_to_concatenated_tiles(config):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        print(f"Created output directory: {config.output_dir}")

    if not os.path.exists(config.input_png_dir):
        print(f"Error: Input PNG directory '{config.input_png_dir}' not found.")
        return

    # ❗使用 config.train_csv_name
    image_list_csv_path = os.path.join(config.data_dir, config.train_csv_name)
    if not os.path.exists(image_list_csv_path):
        print(f"Error: Image list CSV '{config.train_csv_name}' not found at {image_list_csv_path}.")
        return

    df = pd.read_csv(image_list_csv_path)

    print(f"Processing {len(df)} images listed in {config.train_csv_name} from source PNGs in {config.input_png_dir}.")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing source PNGs"):
        image_id = row.image_id # 假設CSV中有 'image_id' 欄位
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

            concatenated_img_pil = Image.fromarray(concatenated_image_np) # astype(np.uint8) 已在拼接函數中處理
            output_filename = os.path.join(config.output_dir, f"{image_id}.png")
            concatenated_img_pil.save(output_filename)

            # 釋放內存
            del source_pil_image, source_np_image, tiles_np_array, concatenated_image_np, concatenated_img_pil
            if (index + 1) % 100 == 0: # 每處理100張圖片執行一次垃圾回收
                gc.collect()

        except UnidentifiedImageError:
            print(f"Error: Cannot identify image file {source_png_path}. It might be corrupted or not a valid image. Skipping.")
        except Exception as e:
            print(f"Error occurred while processing source PNG {source_png_path}: {e}")
            import traceback
            traceback.print_exc() # 打印詳細的錯誤追蹤信息

    print(f"All images from {config.train_csv_name} processed and tiles concatenated into {config.output_dir}!")

# --- Run the script ---
if __name__ == "__main__":
    config = PreprocessConfig()

    # --- ❗用戶配置區 ---
    # 1. 決定使用哪個CSV文件來獲取image_id列表
    # 如果你有 clean_noise.csv 並且想只處理這些圖像，請設置為 "clean_noise.csv"
    config.train_csv_name = "clean_noise.csv" # 或者保持 "train.csv"

    # 2. 設置輸入源PNG的路徑
    config.input_png_dir = "/home/gpl_homee/CVDL_Final/data/train_png" # 你的源PNG文件夾

    # 3. 設置拼接後圖塊PNG的輸出路徑 (建議用新名稱區分不同處理版本)
    config.output_dir = "/home/gpl_homee/CVDL_Final/data/train_concatenated_png_64x128_v2" # 示例：64個128x128圖塊，v2版本，來自cleaned_csv

    # 4. 設置圖塊參數
    config.grid_side_count = 6  # 例如 8x8 = 64 個圖塊
    config.tile_size = 256     # 每個圖塊的大小 128x128
    # --- 結束用戶配置區 ---

    config.num_tiles_total = config.grid_side_count ** 2 # 自動計算總圖塊數

    print(f"Using image list from: {os.path.join(config.data_dir, config.train_csv_name)}")
    print(f"Reading source PNGs from: {config.input_png_dir}")
    print(f"Each source PNG will be processed to extract {config.num_tiles_total} ({config.grid_side_count}x{config.grid_side_count}) 'best' tiles of size {config.tile_size}x{config.tile_size}.")
    print(f"Selected tiles will be concatenated into one image and saved to: {config.output_dir}")
    print(f"Foreground threshold for tile selection: {config.foreground_threshold}")

    process_pngs_to_concatenated_tiles(config)

    # preprocess_png_to_concatenated_tiles_v2.py