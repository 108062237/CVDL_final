import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm # 保持使用 notebook 版本的 tqdm

# --- 從你的 dataloader.py 中導入必要的組件 ---
# 假設 dataloader.py 與此腳本在同一目錄，或者在PYTHONPATH中
# 如果不在同一目錄，你需要調整導入路徑
from dataloader import CFG, PandaDataset, get_transforms # 移除了 set_seed 和 get_dataloaders_for_fold，因為我們直接用全數據

# --- 從你的 train.py 中導入模型定義 ---
# 同樣，假設 train.py 與此腳本在同一目錄，或者 PANDA_Model_Binned 已在此處定義
# 為了簡潔，我直接複製 PANDA_Model_Binned 的定義
import torchvision.models as models
import torch.nn as nn

class PANDA_Model_Binned(nn.Module):
    def __init__(self, base_model_name='efficientnet_b1', num_classes_out=5):
        super(PANDA_Model_Binned, self).__init__()
        if base_model_name.startswith('efficientnet_b1'):
            self.model = models.efficientnet_b1(weights=None)
        elif base_model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=None)
        elif base_model_name == 'efficientnet_b5':
            self.model = models.efficientnet_b5(weights=None)
        else:
            raise ValueError(f"Unsupported base_model_name for PANDA_Model_Binned: {base_model_name}")
        
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential) and len(self.model.classifier) > 1:
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes_out)
        elif hasattr(self.model, '_fc'):
            in_features = self.model._fc.in_features
            self.model._fc = nn.Linear(in_features, num_classes_out)
        else:
            raise AttributeError(f"Could not find a suitable classifier/fc layer in {base_model_name}")

    def forward(self, x):
        return self.model(x)

def collect_oof_predictions(cfg, model_configs, device='cuda'):
    """
    使用多個模型對全部訓練數據進行預測，並收集連續的預測分數。

    Args:
        cfg: CFG類的實例，包含配置信息。
        model_configs: 一個列表，每個元素是一個元組 (模型標籤, 權重路徑, 基礎模型名)。
        device: 'cuda' 或 'cpu'。

    Returns:
        pandas.DataFrame: 包含 image_id, 真實isup_grade, 以及每個模型預測的連續分數。
    """
    print("--- Starting OOF Prediction Collection ---")

    # 1. 加載所有模型
    loaded_models = {}
    for model_label, model_path, model_base in model_configs:
        print(f"Loading {model_label} (base: {model_base})...")
        if os.path.exists(model_path):
            try:
                model = PANDA_Model_Binned(base_model_name=model_base, num_classes_out=cfg.model_output_dim)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.to(device)
                model.eval()
                loaded_models[model_label] = model
                print(f"{model_label} loaded from: {model_path}")
            except Exception as e:
                print(f"Error loading {model_label} from {model_path}: {e}. Skipping.")
        else:
            print(f"Warning: {model_label} weight file not found at {model_path}")

    if not loaded_models:
        print("No models were loaded. Exiting.")
        return pd.DataFrame()

    # 2. 準備數據集和 DataLoader (使用全部 cleaned_train_csv_name 數據)
    full_df_path = os.path.join(cfg.data_dir, cfg.cleaned_train_csv_name)
    if not os.path.exists(full_df_path):
        raise FileNotFoundError(f"Cleaned data CSV '{cfg.cleaned_train_csv_name}' not found at {full_df_path}.")
    
    all_train_df = pd.read_csv(full_df_path)
    all_train_df = all_train_df.dropna(subset=['image_id', cfg.target_col]) # 確保關鍵列無缺失
    all_train_df[cfg.target_col] = all_train_df[cfg.target_col].astype(int)
    print(f"Loaded {len(all_train_df)} samples from {cfg.cleaned_train_csv_name} for prediction.")

    # 使用驗證/測試時的 transform (不進行隨機增強)
    eval_transform = get_transforms(mode="valid", mean=cfg.mean, std=cfg.std)

    full_dataset = PandaDataset(
        df=all_train_df,
        image_dir=cfg.preprocessed_image_dir, # 使用預處理好的拼接圖路徑
        target_col=cfg.target_col,
        transform=eval_transform,
        model_output_dim=cfg.model_output_dim
    )
    
    # batch_size 可以適當調大以加速推斷，num_workers 也可以調整
    # 注意：這裡的 batch_size 和 num_workers 可能與 CFG 中用於訓練的不同，可以單獨設置
    eval_batch_size = getattr(cfg, 'eval_batch_size', cfg.batch_size * 2) # 嘗試用更大batch
    eval_num_workers = getattr(cfg, 'eval_num_workers', max(1, os.cpu_count() // 2 if os.cpu_count() else 1) )

    full_loader = DataLoader(
        full_dataset,
        batch_size=eval_batch_size,
        shuffle=False, # 推斷時不需要打亂
        num_workers=eval_num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # 3. 進行預測並收集結果
    # 初始化一個 DataFrame 來存儲結果
    # 我們需要 image_id 和真實的 isup_grade
    results_df = all_train_df[['image_id', cfg.target_col]].copy()
    results_df.rename(columns={cfg.target_col: 'true_isup_grade'}, inplace=True)


    with torch.no_grad():
        for model_label, model in loaded_models.items():
            print(f"Predicting with {model_label}...")
            model_predictions_continuous = []
            # image_ids_for_model = [] # 如果需要按順序對應，loadershuffle=False很重要

            for images, _ in tqdm(full_loader, desc=f"Predicting ({model_label})"): # 我們不需要 DataLoader 返回的標籤
                images = images.to(device, dtype=torch.float)
                outputs = model(images) # Logits, shape (batch_size, 5)
                
                # 計算連續分數 (sigmoid().sum() 在 round() 之前)
                continuous_scores = outputs.sigmoid().sum(dim=1).cpu().numpy()
                model_predictions_continuous.extend(continuous_scores)
                # image_ids_for_model.extend(batch_image_ids) # 如果DataLoader也返回image_id

            # 確保預測數量與 DataFrame 行數一致
            if len(model_predictions_continuous) == len(results_df):
                results_df[f'pred_continuous_{model_label}'] = model_predictions_continuous
            else:
                print(f"Warning: Mismatch in prediction count for {model_label}. Expected {len(results_df)}, got {len(model_predictions_continuous)}. Skipping this model's predictions.")

    print("--- OOF Prediction Collection Finished ---")
    return results_df

if __name__ == '__main__':
    cfg = CFG()
    # 確保你的模型路徑是正確的，並且相對於你執行此腳本的位置
    # 或者使用絕對路徑
    # 你的模型權重文件路徑 (相對於執行腳本的目錄，或者你需要提供絕對路徑)
    # 這些路徑是基於你提供的 "CVDL_Final/" 結構，假設此腳本在 CVDL_Final 的父目錄中執行
    # 或者，如果此腳本在 CVDL_Final 內，路徑應為 "./PANDA_models_foldX..."
    # 為了保險，最好在運行前確認這些路徑的有效性

    # 假設此腳本與 CVDL_Final 文件夾在同一級別，或者 CVDL_Final 在 PYTHONPATH 中
    # 這裡我假設你的項目根目錄是包含 CVDL_Final 文件夾的目錄
    # 如果不是，你需要調整 base_path
    project_base_path = "." # 或者你 CVDL_Final 的實際父路徑

    model_configurations = [
        ("fold4_b1", os.path.join(project_base_path, "CVDL_Final/PANDA_models_fold4_model_efficientnet_b1/best_model_efficientnet_b1_0.9442.pth"), 'efficientnet_b1'),
        ("fold2_b1", os.path.join(project_base_path, "CVDL_Final/PANDA_models_fold2_model_efficientnet_b1/best_model_efficientnet_b1_0.9483.pth"), 'efficientnet_b1'),
        ("fold1_b1", os.path.join(project_base_path, "CVDL_Final/PANDA_models_fold1_model_efficientnet_b1/best_model_efficientnet_b1_0.9374.pth"), 'efficientnet_b1'),
        ("fold0_b1_v2", os.path.join(project_base_path, "CVDL_Final/PANDA_models_fold0_model_efficientnet_b1/best_model_efficientnet_b1_v2_0.9360.pth"), 'efficientnet_b1'),
        # 如果你有之前提到的B0和B5模型，也可以按照這個格式加進來
        # ("my_b0_model", "/path_to_your_b0_weights/best_model_efficientnet_b0_....pth", 'efficientnet_b0'),
        # ("my_b5_model", "/path_to_your_b5_weights/best_model_efficientnet_b5_....pth", 'efficientnet_b5'),
    ]

    # 檢查模型文件是否存在，只保留存在的
    valid_model_configs = []
    for label, path, base in model_configurations:
        if os.path.exists(path):
            valid_model_configs.append((label, path, base))
        else:
            print(f"Path not found, skipping model {label}: {path}")
    
    if not valid_model_configs:
        print("No valid model paths found. Please check your model_configurations.")
    else:
        # 設置設備
        selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {selected_device}")

        # 收集預測
        oof_results_df = collect_oof_predictions(cfg, valid_model_configs, device=selected_device)

        if not oof_results_df.empty:
            print("\nCollected OOF Predictions (first 5 rows):")
            print(oof_results_df.head())

            # 你可以將這個 DataFrame 保存到 CSV 文件
            output_csv_path = os.path.join(cfg.data_dir, "oof_continuous_predictions.csv")
            oof_results_df.to_csv(output_csv_path, index=False)
            print(f"\nOOF predictions saved to: {output_csv_path}")

            # 接下來，你可以對這個 oof_results_df 進行分析：
            # 1. 計算每個模型單獨使用其連續預測配合優化捨入後的QWK分數。
            # 2. 將多個模型的 pred_continuous_MODEL 列進行平均或加權平均，得到集成後的連續預測。
            # 3. 對集成後的連續預測使用優化捨入，計算最終的OOF QWK。
            # 例如，簡單平均所有模型的連續預測：
            pred_cols = [col for col in oof_results_df.columns if col.startswith('pred_continuous_')]
            if pred_cols:
                oof_results_df['ensemble_continuous_pred'] = oof_results_df[pred_cols].mean(axis=1)
                print("\nEnsemble continuous predictions (first 5 rows):")
                print(oof_results_df[['image_id', 'true_isup_grade', 'ensemble_continuous_pred']].head())
                
                # 這裡可以接續你的優化捨入邏輯
                # from sklearn.metrics import cohen_kappa_score # 需要導入
                # best_thresholds = ... (從 oof_results_df['ensemble_continuous_pred'] 和 oof_results_df['true_isup_grade'] 學習)
                # final_oof_preds = apply_thresholds_to_scores(oof_results_df['ensemble_continuous_pred'].values, best_thresholds)
                # final_oof_qwk = cohen_kappa_score(oof_results_df['true_isup_grade'].values, final_oof_preds, weights='quadratic')
                # print(f"\nEstimated OOF QWK for ensemble (with learned thresholds): {final_oof_qwk}")

            else:
                print("No prediction columns found to create ensemble prediction.")
        else:
            print("OOF prediction collection resulted in an empty DataFrame.")