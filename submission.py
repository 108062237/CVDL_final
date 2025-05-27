# ========================
# 📦 Import Cell
# ========================
import os
import gc
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

import openslide
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

print("Import finish")

# --- 模型定義 ---
class PANDA_Model_Binned(nn.Module):
    def __init__(self, base_model_name='efficientnet_b1', num_classes_out=5):
        super(PANDA_Model_Binned, self).__init__()
        if base_model_name.startswith('efficientnet_b1'):
            self.model = models.efficientnet_b1(weights=None)
        elif base_model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=None)
        # 你可以在這裡根據需要添加對更多基礎模型的支持，例如 b5
        # elif base_model_name == 'efficientnet_b5':
        #     self.model = models.efficientnet_b5(weights=None)
        else:
            raise ValueError(f"Unsupported base_model_name for PANDA_Model_Binned: {base_model_name}")
        
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential) and len(self.model.classifier) > 1:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes_out)
        elif hasattr(self.model, '_fc'): # 兼容其他可能的EfficientNet實現風格
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, num_classes_out)
        else:
            raise AttributeError(f"Could not find a suitable classifier/fc layer in {base_model_name}")

    def forward(self, x):
        return self.model(x)

# --- 加載你的 BCE 模型 ---
# ✨ 更新：包含所有你想集成的 BCE 模型的路徑 ✨
bce_model_paths_and_bases = [
    ("model4 (BCE)", "/kaggle/input/checkpoint/best_model_efficientnet_b1_0.9374.pth", 'efficientnet_b1'),
    ("model5 (BCE)", "/kaggle/input/checkpoint/best_model_efficientnet_b1_0.9483.pth", 'efficientnet_b1'),
    ("model6 (BCE_v2)", "/kaggle/input/checkpoint/best_model_efficientnet_b1_v2_0.9360.pth", 'efficientnet_b1'),
    ("model7 (BCE)", "/kaggle/input/checkpoint2/best_model_efficientnet_b1_0.9442.pth", 'efficientnet_b1'),
    ("model_new_b1 (BCE)", "/kaggle/input/checkpoint6/best_model_efficientnet_b1_0.9344.pth", 'efficientnet_b1'), # ✨ 新增的模型 ✨
]

active_bce_models = []
for model_label, model_path, model_base in bce_model_paths_and_bases:
    print(f"Loading {model_label}...")
    if os.path.exists(model_path):
        try:
            current_model = PANDA_Model_Binned(base_model_name=model_base, num_classes_out=5)
            current_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            current_model.eval()
            active_bce_models.append(current_model)
            print(f"{model_label} loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading {model_label} from {model_path}: {e}. Skipping.")
    else:
        print(f"Warning: {model_label} weight file not found at {model_path}")

# --- 配置, 圖塊處理, Dataset (與你之前的版本一致) ---
class CFG:
    data_dir = '../input/prostate-cancer-grade-assessment'
    tile_size = 256; n_tiles = 36; batch_size = 8
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]

def get_tiles(img_np, tile_size=CFG.tile_size, n_tiles=CFG.n_tiles):
    h, w, c = img_np.shape; pad_h = (tile_size - h % tile_size) % tile_size; pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h > 0 or pad_w > 0: img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)
    h_padded, w_padded, _ = img_np.shape
    img_reshaped = img_np.reshape(h_padded // tile_size, tile_size, w_padded // tile_size, tile_size, c)
    img_transposed = img_reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, c)
    if len(img_transposed) == 0: tiles = np.array([])
    else: tile_means = img_transposed.reshape(img_transposed.shape[0], -1).mean(axis=-1); idxs = np.argsort(tile_means)[:n_tiles]; tiles = img_transposed[idxs]
    if len(tiles) < n_tiles:
        white_tile = np.full((tile_size, tile_size, c), 255, dtype=np.uint8); white_tiles_needed = n_tiles - len(tiles)
        if white_tiles_needed > 0:
            padding_tiles = np.stack([white_tile] * white_tiles_needed, axis=0)
            if len(tiles) == 0: tiles = padding_tiles
            else: tiles = np.concatenate([tiles, padding_tiles], axis=0)
    tiles = tiles[:n_tiles]; return tiles

def tile_concat(tiles, tile_size=CFG.tile_size, n_tiles=CFG.n_tiles):
    grid_size = int(n_tiles ** 0.5)
    if grid_size * grid_size != n_tiles: print(f"Warning: n_tiles ({n_tiles}) is not a perfect square for {grid_size}x{grid_size} grid.")
    channels = tiles.shape[-1] if tiles.ndim == 4 and tiles.shape[0] > 0 else 3
    result = np.zeros((tile_size * grid_size, tile_size * grid_size, channels), dtype=np.uint8)
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < len(tiles): result[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tiles[idx]
            else: result[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = np.full((tile_size,tile_size,channels),255,dtype=np.uint8)
    return result

image_folder = os.path.join(CFG.data_dir, 'test_images')
submission_csv_path = os.path.join(CFG.data_dir, 'sample_submission.csv')

if not os.path.exists(image_folder) or not os.listdir(image_folder):
    print(f"Test image folder '{image_folder}' not found or empty. Using train_images for local testing.")
    image_folder = os.path.join(CFG.data_dir, 'train_images')
    is_test_environment = False
    train_csv_path = os.path.join(CFG.data_dir, 'train.csv')
    if os.path.exists(train_csv_path):
        df_all_train = pd.read_csv(train_csv_path)
        if 'image_id' in df_all_train.columns: test_df = df_all_train[['image_id']].sample(min(16,len(df_all_train)),random_state=42)
        else: test_df = pd.DataFrame(columns=['image_id'])
    else: test_df = pd.DataFrame(columns=['image_id'])
else:
    is_test_environment = True
    if os.path.exists(submission_csv_path):
        test_df = pd.read_csv(submission_csv_path)
        if 'isup_grade' in test_df.columns: # sample_submission might have it
             test_df = test_df[['image_id']] # Only keep image_id for prediction
    elif os.path.isdir(image_folder): # Fallback to listing files if sample_submission not found
        print(f"Warning: '{submission_csv_path}' not found. Listing image_ids from directory '{image_folder}'.")
        test_df = pd.DataFrame()
        test_df['image_id'] = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith('.tiff')]
    else:
        print(f"Error: Test image folder '{image_folder}' is not a valid directory and sample_submission.csv not found.")
        test_df = pd.DataFrame(columns=['image_id'])


print(f"Inferencing on image folder: {image_folder}. Number of images: {len(test_df)}")

class PandaSubmissionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, tile_size=CFG.tile_size, n_tiles=CFG.n_tiles):
        self.df = df.reset_index(drop=True); self.image_dir = image_dir; self.transform = transform
        self.tile_size = tile_size; self.n_tiles = n_tiles
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        if self.df.empty or index >= len(self.df):
            grid_dim = int(self.n_tiles**0.5) if self.n_tiles > 0 else 1
            placeholder_np = np.full((self.tile_size*grid_dim,self.tile_size*grid_dim,3),255,dtype=np.uint8)
            image = Image.fromarray(placeholder_np); image_id = "ERROR_LOADING_IMAGE_ID" # Use a distinct ID for errors
            if self.transform: image = self.transform(image)
            return image, image_id
        row = self.df.iloc[index]; image_id = row.image_id; slide_path = os.path.join(self.image_dir, f"{image_id}.tiff")
        image_concatenated_np = None
        try:
            if not os.path.exists(slide_path): raise FileNotFoundError(f"FATAL: Test slide file not found: {slide_path}")
            slide = openslide.OpenSlide(slide_path)
            target_level = 1 if len(slide.level_dimensions) > 1 else 0
            page1_pil = slide.read_region((0,0),target_level,slide.level_dimensions[target_level]).convert("RGB")
            page1_np = np.array(page1_pil); slide.close()
            tiles = get_tiles(page1_np,self.tile_size,self.n_tiles)
            image_concatenated_np = tile_concat(tiles,self.tile_size,self.n_tiles)
            image = Image.fromarray(image_concatenated_np)
        except Exception as e:
            print(f"Error processing TEST image {slide_path}: {e}. Using placeholder image.")
            grid_dim = int(self.n_tiles**0.5) if self.n_tiles > 0 else 1
            placeholder_np = np.full((self.tile_size*grid_dim,self.tile_size*grid_dim,3),255,dtype=np.uint8)
            image = Image.fromarray(placeholder_np)
        if self.transform: image = self.transform(image)
        return image, image_id

submission_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=CFG.mean,std=CFG.std)])

if test_df.empty and 'image_id' not in test_df.columns: # Ensure test_df has 'image_id' even if empty
    test_df = pd.DataFrame(columns=['image_id'])
if test_df.empty: # If still empty after all attempts
    print("Warning: test_df is empty before creating Dataset. Submission will likely be problematic.")
    if is_test_environment : # Kaggle環境
        test_df = pd.DataFrame({'image_id': ['KAGGLE_TEST_EMPTY_IMAGE_ID_FALLBACK']}) # 提交時至少有一行
    else: # 本地測試
        test_df = pd.DataFrame({'image_id': ['local_dummy_id_fallback']})


submission_dataset = PandaSubmissionDataset(test_df, image_dir=image_folder, transform=submission_transform)
num_loader_workers = max(1, os.cpu_count() // 2 if os.cpu_count() else 1)
submission_loader = DataLoader(submission_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=num_loader_workers)

# --- 推論函數 ---
def inference_ensemble(bce_models_list, loader, device='cuda'):
    for model_instance in bce_models_list:
        if model_instance is not None:
            model_instance.to(device)
            model_instance.eval()

    image_ids_all = []
    final_pred_labels_all = []
    
    # 確保 loader.dataset.df['image_id'] 的 image_id 列表在循環外獲取一次，以應對 loader 為空的情況
    # 但 loader 為空，image_ids 也應該為空，所以主要處理是 pred_labels_all
    expected_image_ids = []
    if hasattr(loader.dataset, 'df') and not loader.dataset.df.empty and 'image_id' in loader.dataset.df.columns:
        expected_image_ids = loader.dataset.df['image_id'].tolist()

    if len(loader) == 0 or not bce_models_list or all(m is None for m in bce_models_list):
        print("Warning: DataLoader is empty or no valid BCE models to ensemble. No predictions will be made.")
        final_pred_labels_all = np.zeros(len(expected_image_ids), dtype=int).tolist() # 使用期望的長度
        return expected_image_ids, final_pred_labels_all # 返回期望的 image_ids 和默認預測

    with torch.no_grad():
        for batch_idx, (images, image_ids_batch) in enumerate(tqdm(loader, desc="Submission Inference")): # batch 的 image_ids
            images = images.to(device, dtype=torch.float)
            
            all_batch_outputs = []
            for model_instance in bce_models_list:
                if model_instance is not None:
                    outputs_bce = model_instance(images)
                    all_batch_outputs.append(outputs_bce)
            
            if not all_batch_outputs:
                print(f"Warning: No models produced output for batch {batch_idx}. Filling with default predictions (0).")
                batch_final_preds = np.zeros(len(image_ids_batch), dtype=int)
            else:
                stacked_outputs = torch.stack(all_batch_outputs, dim=0)
                avg_outputs = torch.mean(stacked_outputs.float(), dim=0)
                batch_final_preds = avg_outputs.sigmoid().sum(dim=1).round().cpu().numpy().astype(int)
        
            final_pred_labels_all.extend(batch_final_preds)
            image_ids_all.extend(image_ids_batch) # 使用來自 DataLoader 的 image_ids
            
    # 推論結束後，再次校驗 image_ids_all 是否與 expected_image_ids 匹配（如果需要嚴格順序）
    # 但通常 DataLoader (shuffle=False) 會保持順序
    if len(image_ids_all) != len(expected_image_ids) and expected_image_ids:
        print(f"Warning: Number of collected image_ids ({len(image_ids_all)}) differs from initially expected ({len(expected_image_ids)}). This might indicate an issue.")
        # 如果 image_ids_all 為空但 expected_image_ids 不為空，則用 expected_image_ids
        if not image_ids_all and expected_image_ids:
            image_ids_all = expected_image_ids
            if len(final_pred_labels_all) != len(image_ids_all): # 如果預測也為空或長度不對
                final_pred_labels_all = np.zeros(len(image_ids_all), dtype=int).tolist()


    return image_ids_all, final_pred_labels_all

# --- 主執行部分 ---
if not active_bce_models: # 使用 active_bce_models
    print("Error: No models were successfully loaded for the ensemble. Creating an empty or default submission.")
    submission_ids = submission_df_for_loader['image_id'].tolist() if 'image_id' in submission_df_for_loader.columns and not submission_df_for_loader.empty else ['no_images_loaded_for_submission']
    submission_preds = np.zeros(len(submission_ids), dtype=int).tolist()
    
    submission = pd.DataFrame({"image_id": submission_ids, "isup_grade": submission_preds})
else:
    print(f"Starting submission inference with an ensemble of {len(active_bce_models)} BCE model(s).") # 使用 active_bce_models
    image_ids, pred_labels = inference_ensemble(active_bce_models, submission_loader) # 使用 active_bce_models

    # 創建 submission DataFrame，優先使用從 loader 中獲取的 image_ids 順序
    # 如果 image_ids 為空但 submission_df_for_loader 不為空，則使用後者的 image_id 列表
    final_submission_ids = image_ids
    if not image_ids and not submission_df_for_loader.empty and 'image_id' in submission_df_for_loader.columns:
        final_submission_ids = submission_df_for_loader['image_id'].tolist()
        if len(pred_labels) != len(final_submission_ids): # 校驗長度
            print(f"Warning: Prediction count ({len(pred_labels)}) does not match ID count ({len(final_submission_ids)}) from submission template. Adjusting predictions to zeros.")
            pred_labels = np.zeros(len(final_submission_ids), dtype=int).tolist()
    elif not image_ids and (submission_df_for_loader.empty or 'image_id' not in submission_df_for_loader.columns):
        # 極端情況，沒有任何 image_id 可以使用
        final_submission_ids = ['ERROR_NO_IMAGE_IDS']
        pred_labels = [0]


    submission = pd.DataFrame({
        "image_id": final_submission_ids,
        "isup_grade": pred_labels
    })

# 儲存結果
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
if not submission.empty:
    print(submission.head())
    if "isup_grade" in submission.columns:
        print("\nPredicted ISUP Grade Distribution:")
        print(submission["isup_grade"].value_counts().sort_index())
    else:
        print("\n'isup_grade' column missing in the submission DataFrame.")
else:
    print("\nSubmission DataFrame is empty.")
