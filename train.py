import os
import time
import argparse  # Used to pass arguments from the command line, such as fold number

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Note: using tqdm, not tqdm.notebook
from sklearn.metrics import cohen_kappa_score
import torchvision.models as models

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
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, current_num_classes)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented yet.")

    def forward(self, x):
        return self.model(x)

# --- Training and validation functions ---
def train_fn(train_loader, model, criterion, optimizer, device, scheduler=None):
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
    if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
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
    model = PandaSimpleModel(model_name='resnet34', pretrained=True, num_classes=config.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    num_epochs_config = getattr(config, 'num_epochs', 10)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_config, eta_min=1e-6)

    best_kappa = -1.0
    output_dir = f"./PANDA_models_fold{current_fold}"
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f"best_model_seed{config.seed}.pth")

    print(f"--- Starting training for Fold {current_fold} ---")
    print(f"Model will be saved at: {output_dir}")
    
    for epoch in range(num_epochs_config):
        start_time = time.time()
        
        train_loss = train_fn(train_loader, model, criterion, optimizer, device, scheduler if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None)
        val_loss, val_kappa = valid_fn(valid_loader, model, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch {epoch+1}/{num_epochs_config} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s")
        print(f"\tTraining Loss: {train_loss:.4f}")
        print(f"\tValidation Loss: {val_loss:.4f} | Validation Kappa: {val_kappa:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\tCurrent Learning Rate: {current_lr:.6f}")

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_kappa)

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
    parser.add_argument('--epochs', type=int, default=10, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (override CFG setting)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of data loading workers (override CFG setting)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (small data)')

    args = parser.parse_args()

    config = CFG()
    config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.debug:
        config.debug = True
        print("DEBUG mode enabled via command line.")

    run_training(config, args.fold)
