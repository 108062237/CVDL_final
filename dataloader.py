# dataloader.py

import os
import random
import numpy as np
import pandas as pd
# import cv2 
# import skimage.io 
from PIL import Image  # Used to convert NumPy array to PIL Image

# import openslide

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from torchvision import transforms 

# --- Configuration ---
class CFG:
    # Basic settings
    competition_name = "PANDA"
    seed = 42
    debug = False  

    model_name = "efficientnet_b1"

    # Data path (raw data)
    data_dir = "/home/gpl_homee/CVDL_Final/data" 
    preprocessed_image_dir = "/home/gpl_homee/CVDL_Final/data/train_concatenated_png_36*256_v2"
    cleaned_train_csv_name = "clean_noise.csv"

    # Image processing parameters
    # tile_size = 256
    # tile_count = 6 
    # n_tiles = tile_count ** 2

    # wsi_level = 1

    # Classification task parameters
    num_classes = 6  # ISUP grade 0~5
    model_output_dim = 5

    # Training parameters
    n_fold = 4
    fold_seed = 42
    target_col = "isup_grade"
    num_workers = 0
    batch_size = 4
    image_format = "png"
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]  

def set_seed(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Dataset class ---
class PandaDataset(Dataset):
    def __init__(self, df, image_dir, target_col, transform=None, model_output_dim=CFG.model_output_dim):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir 
        self.target_col = target_col
        self.transform = transform
        self.model_output_dim = model_output_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row.image_id
        isup_grade = row[self.target_col]

        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        
        try:
            image = Image.open(image_path).convert('RGB') 

        except Exception as e:
            print(f"load {image_path} fail: {e}")
            placeholder_size = getattr(CFG, 'final_image_size', (256*6, 256*6)) 
           
            image_data = np.full((placeholder_size, placeholder_size, 3), 255, dtype=np.uint8)
            image = Image.fromarray(image_data)
            label = 0 


        if self.transform:
            image = self.transform(image) 
        
        label_binned = np.zeros(self.model_output_dim, dtype=np.float32)
        if isup_grade > 0:
            label_binned[:isup_grade] = 1.0

        return image, torch.tensor(label_binned, dtype=torch.float32)

# --- Data augmentation Transforms ---
def get_transforms(mode="train", mean=CFG.mean, std=CFG.std):
    
    if mode == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
    elif mode == "valid" or mode == "test":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

# --- Function to get Dataloaders ---
def get_dataloaders_for_fold(fold_to_run, cfg):
    """
    Prepare Dataloaders for specified fold.
    """
    set_seed(cfg.seed) 

    df_path = os.path.join(cfg.data_dir, cfg.cleaned_train_csv_name)
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"train.csv not found in {cfg.data_dir}.")

    df = pd.read_csv(df_path)

    validation_fold_value_in_csv = fold_to_run + 1 

    print(f"Fold {fold_to_run}: Using rows where 'kfold' == {validation_fold_value_in_csv} in CSV as validation set.")

    train_df = df[df['kfold'] != validation_fold_value_in_csv].reset_index(drop=True)
    valid_df = df[df['kfold'] == validation_fold_value_in_csv].reset_index(drop=True)

    train_df = train_df.dropna(subset=['image_id', cfg.target_col])
    valid_df = valid_df.dropna(subset=['image_id', cfg.target_col])
    train_df[cfg.target_col] = train_df[cfg.target_col].astype(int)
    valid_df[cfg.target_col] = valid_df[cfg.target_col].astype(int)

    if cfg.debug:
        print("DEBUG mode: using small sample.")
        sample_size = max(cfg.batch_size * 2, 10) # 確保樣本量至少為10或夠兩個batch
        train_df = train_df.sample(n=min(len(train_df), sample_size), random_state=cfg.seed).reset_index(drop=True)
        valid_df = valid_df.sample(n=min(len(valid_df), sample_size), random_state=cfg.seed).reset_index(drop=True)

    image_dir_to_use = getattr(cfg, 'preprocessed_image_dir', './default_image_dir_if_not_in_cfg')
    model_output_dim_for_dataset = getattr(cfg, 'model_output_dim', 5)

    train_dataset = PandaDataset(
        df=train_df,
        image_dir=image_dir_to_use,
        target_col=cfg.target_col,
        transform=get_transforms(mode="train", mean=cfg.mean, std=cfg.std),
        model_output_dim=model_output_dim_for_dataset 
    )
    valid_dataset = PandaDataset(
        df=valid_df,
        image_dir=image_dir_to_use,
        target_col=cfg.target_col,
        transform=get_transforms(mode="valid", mean=cfg.mean, std=cfg.std),
        model_output_dim=model_output_dim_for_dataset 
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(), 
        drop_last=True 
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    print(f"Fold {fold_to_run}: Loading images from '{image_dir_to_use}'.")
    if len(train_dataset) > 0 and len(valid_dataset) > 0:
        print(f"{len(train_dataset)} training samples, {len(valid_dataset)} validation samples.")
    else:
        print(f"Warning: Training or validation set is empty. Train: {len(train_dataset)}, Valid: {len(valid_dataset)}.")

    return train_loader, valid_loader

if __name__ == '__main__':
    print("Run dataloader.py directly to test...")

    cfg_instance = CFG()
    cfg_instance.debug = True 
    cfg_instance.num_workers = 0 
    
    current_fold_to_test = 0
    print(f"Testing Dataloader for Fold {current_fold_to_test}...")

    try:
        train_loader_test, valid_loader_test = get_dataloaders_for_fold(current_fold_to_test, cfg_instance)

        print(f"\nSuccessfully built Dataloaders for Fold {current_fold_to_test}.")
        print(f"Train DataLoader has {len(train_loader_test)} batches.")
        print(f"Validation DataLoader has {len(valid_loader_test)} batches.")

        print("\nInspect one batch from training DataLoader:")
        train_images, train_labels = next(iter(train_loader_test))
        print(f"Image shape: {train_images.shape}, Label shape: {train_labels.shape}")
        print(f"Image dtype: {train_images.dtype}, Label dtype: {train_labels.dtype}")
        print(f"Image min/max: {train_images.min().item():.2f}/{train_images.max().item():.2f}")
        print(f"Label sample: {train_labels[:4]}")

        print("\nInspect one batch from validation DataLoader:")
        valid_images, valid_labels = next(iter(valid_loader_test))
        print(f"Image shape: {valid_images.shape}, Label shape: {valid_labels.shape}")

        print("\nDataloader test completed.")
        
    except Exception as e:
        print(f"Error occurred during Dataloader test: {e}")
        import traceback
        traceback.print_exc()
