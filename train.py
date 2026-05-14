import os
import yaml
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChangeDetectionDataset, get_train_transforms, get_val_transforms
from model import DualStreamAttentionUNet


# -------------------------------
# Seed
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------
# Dice Loss
# -------------------------------
class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)[:, 1]
        targets = targets.float()
        intersection = (preds * targets).sum()
        return 1 - (2. * intersection + 1e-8) / (preds.sum() + targets.sum() + 1e-8)


# -------------------------------
# Focal Loss
# -------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        ce = nn.functional.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# -------------------------------
# Metrics
# -------------------------------
def compute_metrics(preds, masks):
    preds = preds.flatten()
    masks = masks.flatten()

    tp = ((preds == 1) & (masks == 1)).sum().item()
    fp = ((preds == 1) & (masks == 0)).sum().item()
    fn = ((preds == 0) & (masks == 1)).sum().item()
    tn = ((preds == 0) & (masks == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return precision, recall, f1, iou, tp, fp, fn, tn


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_tp = total_fp = total_fn = total_tn = 0
    total_loss = 0

    with torch.no_grad():
        for eo, sar, masks in loader:
            eo, sar, masks = eo.to(device), sar.to(device), masks.to(device)

            outputs = model(eo, sar)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            _, _, _, _, tp, fp, fn, tn = compute_metrics(preds, masks)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    return {
        "loss": total_loss / len(loader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }


# -------------------------------
# Training
# -------------------------------
def train():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    set_seed(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🖥️ Using device: {device}")
    if device.type == 'cuda':
        print(f"💾 GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    train_loader = DataLoader(
        ChangeDetectionDataset(
            config['train']['pre_dir'],
            config['train']['post_dir'],
            config['train']['mask_dir'],
            transform=get_train_transforms()
        ),
        batch_size=config['train']['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        ChangeDetectionDataset(
            config['val']['pre_dir'],
            config['val']['post_dir'],
            config['val']['mask_dir'],
            transform=get_val_transforms()
        ),
        batch_size=config['val']['batch_size'],
        shuffle=False
    )

    # Model
    model = DualStreamAttentionUNet(
        eo_channels=config['model']['eo_channels'],
        sar_channels=config['model']['sar_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)

    # Loss (🔥 weighted for better IoU)
    focal = FocalLoss()
    dice = DiceLoss()

    def loss_fn(x, y):
        return 0.5 * focal(x, y) + 0.5 * dice(x, y)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])

    # 🔥 LR Scheduler (important)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, verbose=True
    )

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    best_iou = 0
    patience = 10
    patience_counter = 0

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []

    # -------------------------------
    # Training Loop
    # -------------------------------
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0
        total_tp = total_fp = total_fn = total_tn = 0

        for eo, sar, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            eo, sar, masks = eo.to(device), sar.to(device), masks.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(eo, sar)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            _, _, _, _, tp, fp, fn, tn = compute_metrics(preds, masks)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        train_loss = running_loss / len(train_loader)
        train_precision = total_tp / (total_tp + total_fp + 1e-8)
        train_recall = total_tp / (total_tp + total_fn + 1e-8)
        train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-8)

        val_metrics = evaluate(model, val_loader, device, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        train_f1s.append(train_f1)
        val_f1s.append(val_metrics['f1'])

        print(f"""
Epoch {epoch+1}
Train Loss: {train_loss:.4f}
Val Loss: {val_metrics['loss']:.4f}
Train F1: {train_f1:.4f}
Val F1: {val_metrics['f1']:.4f}
IoU: {val_metrics['iou']:.4f}
Precision: {val_metrics['precision']:.4f}
Recall: {val_metrics['recall']:.4f}
""")

        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        # LR scheduler
        scheduler.step(val_metrics['iou'])

        # Early stopping
        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_f1s, label="Train F1")
    plt.plot(val_f1s, label="Val F1")
    plt.legend()
    plt.title("F1 Score")

    plt.savefig("train_val_curves.png")
    print("🎯 Training complete")
    

if __name__ == "__main__":
    train()