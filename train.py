import os
import time
import argparse  # Used to pass arguments from the command line, such as fold number

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Note: using tqdm, not tqdm.notebook
from sklearn.metrics import cohen_kappa_score
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Import from our custom dataloader.py
from dataloader import CFG, get_dataloaders_for_fold, set_seed

# --- Model definition ---
class PandaSimpleModel(nn.Module):
    def __init__(self, model_name='resnet34', pretrained=True, num_classes=CFG.num_classes):
        super().__init__()
        current_num_classes = num_classes 

        if model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, current_num_classes)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, current_num_classes)

        # --- EfficientNet (torchvision.models) ---
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        elif model_name == 'efficientnet_b1':
            self.model = models.efficientnet_b1(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        elif model_name == 'efficientnet_b2':
            self.model = models.efficientnet_b2(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        elif model_name == 'efficientnet_b3':
            self.model = models.efficientnet_b3(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        elif model_name == 'efficientnet_b4':
            self.model = models.efficientnet_b4(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        elif model_name == 'efficientnet_b5':
            self.model = models.efficientnet_b5(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        elif model_name == 'efficientnet_b6':
            self.model = models.efficientnet_b6(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        elif model_name == 'efficientnet_b7':
            self.model = models.efficientnet_b7(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)

        # --- ResNeXt ---
        elif model_name == 'resnext50_32x4d': 
            self.model = models.resnext50_32x4d(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, current_num_classes)
        elif model_name == 'resnext101_32x8d': 
            self.model = models.resnext101_32x8d(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, current_num_classes)
        elif model_name == 'resnext50_32x4d_ssl': 
            self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, current_num_classes)
        elif model_name == 'resnext101_32x4d_swsl': 
            self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, current_num_classes)
        elif model_name == 'resnext101_32x8d_swsl': 
            self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x8d_swsl')
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, current_num_classes)


        # --- DenseNet ---
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, current_num_classes)
        elif model_name == 'densenet169':
            self.model = models.densenet169(pretrained=pretrained)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, current_num_classes)

        # --- Vision Transformer (ViT) - 需要 timm 庫或較新 torchvision ---
        # 安裝: pip install timm
        # elif model_name == 'vit_base_patch16_224': # 範例，輸入尺寸通常是 224x224
        #     try:
        #         import timm
        #         self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0) # num_classes=0 移除原始頭部
        #         in_features = self.model.head.in_features # ViT 的頭部通常是 self.model.head
        #         self.model.head = nn.Linear(in_features, current_num_classes) # 加上我們自己的頭部
        #     except ImportError:
        #         raise ImportError("請安裝 timm 庫以使用 Vision Transformer: pip install timm")
        #     except Exception as e: # timm 模型結構可能變化
        #         print(f"載入 timm 模型 {model_name} 出錯: {e}")
        #         print("嘗試通用 ViT 頭部結構...")
        #         # 通用 ViT 結構 (可能需要根據具體模型調整)
        #         # 假設 self.model 已經載入且移除了原始頭部
        #         # 這裡需要知道 ViT 在移除頭部後的輸出特徵維度
        #         # 例如，對於 'vit_base_patch16_224'，通常是 768
        #         # in_features = 768 # 根據所選 ViT 模型設定
        #         # self.model.head = nn.Linear(in_features, current_num_classes) # 假設頭部名為 head
        #         # pass # 您需要根據具體的 ViT 模型來確定如何獲取 in_features

        else:
            raise NotImplementedError(f"model {model_name} is not implement or incorrect name.")

    def forward(self, x):
        return self.model(x)

# --- Training and validation functions ---
def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=True, position=0)
    for images, labels in progress_bar:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(valid_loader, desc="Validation", leave=True, position=0)
    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    epoch_loss = running_loss / len(valid_loader.dataset)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return epoch_loss, kappa

# --- Main execution function ---
def run_training(config, current_fold):
    set_seed(config.seed)

    print(f"======== Initializing training for Fold {current_fold} ========")
    
    print("Loading data...")
    train_loader, valid_loader = get_dataloaders_for_fold(current_fold, config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    # Ensure config has model_name attribute or provide a default
    model_name_to_use = getattr(config, 'model_name', 'efficientnet_b0') # Default to efficientnet_b0
    model = PandaSimpleModel(model_name=model_name_to_use, pretrained=True, num_classes=config.num_classes)
    model.to(device)
    print(f"Using model: {model_name_to_use}")


    if hasattr(train_loader.dataset, 'df') and config.target_col in train_loader.dataset.df.columns:
        labels_for_weights = train_loader.dataset.df[config.target_col].values
        class_labels_unique = np.unique(labels_for_weights)

        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=class_labels_unique,
            y=labels_for_weights
        )
        class_weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)
        print(f"Class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Warning: Could not access training labels to compute class weights. Using unweighted CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer Settings
    initial_lr = getattr(config, 'lr', 3e-5) 
    weight_decay_val = getattr(config, 'weight_decay', 1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay_val)
    print(f"Optimizer: AdamW, Initial LR: {initial_lr:.2e}, Weight Decay: {weight_decay_val:.1e}")
    
    # Learning Rate Scheduler Settings
    num_epochs_total = getattr(config, 'num_epochs', 15) # Total epochs from config
    warmup_epochs = getattr(config, 'warmup_epochs', 3)   # Warmup epochs from config, default 3
    
    scheduler_cosine = None # Initialize to None
    if warmup_epochs >= num_epochs_total:
        print(f"Warning: Warmup epochs ({warmup_epochs}) is >= total epochs ({num_epochs_total}). No CosineAnnealingLR will be used after warmup.")
    else:
        cosine_t_max = num_epochs_total - warmup_epochs
        if cosine_t_max <= 0: # Should not happen if warmup_epochs < num_epochs_total
             print(f"Error: cosine_t_max ({cosine_t_max}) is not positive. Check epoch configurations.")
             cosine_t_max = 1 # Fallback to avoid error, but indicates config issue
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=getattr(config, 'eta_min', 1e-5))
        print(f"Learning Rate Scheduler: Warmup ({warmup_epochs} epochs) + CosineAnnealingLR (T_max={cosine_t_max}, eta_min={getattr(config, 'eta_min', 1e-5):.1e})")

    best_kappa = -1.0
    output_dir = f"./PANDA_models_fold{current_fold}"
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"best_model_seed{config.seed}.pth")

    print(f"--- Starting training for Fold {current_fold} ---")
    print(f"Models will be saved to: {output_dir}")
    
    for epoch in range(num_epochs_total):
        start_time = time.time()
        
        # --- Warmup Logic ---
        if epoch < warmup_epochs:
            # Simple linear warmup from a smaller fraction of initial_lr to initial_lr
            # Or from a very small lr (e.g., initial_lr / 10) to initial_lr
            # Example: lr starts at initial_lr/10 and ramps up
            # lr_start_warmup = initial_lr / 10 
            # current_epoch_lr = lr_start_warmup + (initial_lr - lr_start_warmup) * (epoch + 1) / warmup_epochs
            # A simpler linear ramp from 0 (or very small) to initial_lr:
            current_epoch_lr = initial_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_epoch_lr
            print(f"Warmup Epoch {epoch+1}/{warmup_epochs}: Setting LR to {current_epoch_lr:.2e}")
        elif epoch == warmup_epochs: 
            # End of warmup, ensure LR is set to the initial_lr for CosineAnnealingLR to start correctly
            print(f"Warmup finished. Setting LR to initial value: {initial_lr:.2e} for CosineAnnealingLR.")
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        # Pass scheduler_step_per_epoch=False during warmup, as CosineAnnealingLR is stepped later
        train_loss = train_fn(train_loader, model, criterion, optimizer, device)
        val_loss, val_kappa = valid_fn(valid_loader, model, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch {epoch+1}/{num_epochs_total} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tValid Loss: {val_loss:.4f} | Valid Kappa: {val_kappa:.4f}")
        
        # --- Scheduler Step (after warmup) ---
        if scheduler_cosine and epoch >= warmup_epochs:
            scheduler_cosine.step()
            
        current_lr_display = optimizer.param_groups[0]['lr']
        print(f"\tCurrent Learning Rate: {current_lr_display:.6f}")


        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), best_model_path)
            print(f"\tSaved new best model! Kappa: {best_kappa:.4f} (at Epoch {epoch+1})")
            print(f"\tModel saved to: {best_model_path}")
        
    print(f"--- Fold {current_fold} training completed ---")
    print(f"Best Validation Kappa: {best_kappa:.4f}")
    print(f"Best model saved to: {best_model_path}")
    
    return best_kappa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PANDA Challenge Training Script")
    parser.add_argument('--fold', type=int, default=0, help='Fold number to train (0 to n_fold-1)')
    parser.add_argument('--epochs', type=int, default=15, help='Total number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides CFG)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers (overrides CFG)')
    parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate') # Defaulting to 3e-5
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs') # Defaulting to 3
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help='Name of the model backbone to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (uses a small subset of data)')

    args = parser.parse_args()

    config = CFG() # Load defaults from dataloader.py
    config.num_epochs = args.epochs
    config.warmup_epochs = args.warmup_epochs # Add warmup_epochs to config
    config.lr = args.lr # Add lr to config
    config.model_name = args.model_name # Add model_name to config

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.debug:
        config.debug = True
        print("Debug mode enabled by command line argument.")
    
    run_training(config, args.fold)
