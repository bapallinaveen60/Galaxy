import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2

from dataset import ChangeDetectionDataset, get_val_transforms
from model import DualStreamAttentionUNet   # ✅ FIXED


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

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


# -------------------------------
# Visualization
# -------------------------------
def visualize_predictions(eo, sar, mask, pred, save_path, idx):
    eo = eo.cpu().numpy().transpose(1, 2, 0)
    sar = sar.cpu().numpy().transpose(1, 2, 0)
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    eo_vis = (eo - eo.min()) / (eo.max() - eo.min() + 1e-8)
    sar_vis = sar.squeeze()
    sar_vis = (sar_vis - sar_vis.min()) / (sar_vis.max() - sar_vis.min() + 1e-8)

    overlay = eo_vis.copy()
    overlay[pred == 1] = [1, 0, 0]

    fig, axs = plt.subplots(1, 5, figsize=(18, 4))

    axs[0].imshow(eo_vis)
    axs[0].set_title("EO")

    axs[1].imshow(sar_vis, cmap='gray')
    axs[1].set_title("SAR")

    axs[2].imshow(mask, cmap='gray')
    axs[2].set_title("Ground Truth")

    axs[3].imshow(pred, cmap='gray')
    axs[3].set_title("Prediction")

    axs[4].imshow(overlay)
    axs[4].set_title("Overlay")

    for ax in axs:
        ax.axis('off')

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"sample_{idx}.png"))
    plt.close()


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model, loader, device, save_vis=False, vis_dir="outputs/visualizations"):
    model.eval()
    total = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    with torch.no_grad():
        for i, (eo, sar, masks) in enumerate(loader):
            eo = eo.to(device)
            sar = sar.to(device)
            masks = masks.to(device)

            # ✅ FIXED (no concat)
            logits = model(eo, sar)

            probs = torch.softmax(logits, dim=1)[:, 1]
            threshold = 0.5
            preds = (probs > threshold).long()

            # Morphological filtering (per sample in batch)
            pred_np = preds.cpu().numpy().astype(np.uint8)
            kernel = np.ones((3,3), np.uint8)
            for b in range(pred_np.shape[0]):
                pred_np[b] = cv2.morphologyEx(pred_np[b], cv2.MORPH_OPEN, kernel)
            preds = torch.from_numpy(pred_np).to(device)

            metrics = compute_metrics(preds, masks)
            for k in total:
                total[k] += metrics[k]

            if save_vis and i < 10:
                for b in range(eo.shape[0]):
                    visualize_predictions(
                        eo[b], sar[b], masks[b], preds[b],
                        save_path=vis_dir,
                        idx=i * loader.batch_size + b
                    )

    precision = total['tp'] / (total['tp'] + total['fp'] + 1e-8)
    recall = total['tp'] / (total['tp'] + total['fn'] + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = total['tp'] / (total['tp'] + total['fp'] + total['fn'] + 1e-8)

    total.update({
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    })

    return total


# -------------------------------
# Confusion Matrix Heatmap
# -------------------------------
def plot_confusion_matrix(tp, fp, fn, tn, save_path="outputs/visualizations"):
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    cm_normalized = cm / cm.sum() * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=11)
    
    # Add labels and annotations
    labels = ['No Change', 'Change']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]:.0f}\n({cm_normalized[i, j]:.1f}%)',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


# -------------------------------
# Metrics Heatmap
# -------------------------------
def plot_metrics_heatmap(metrics, save_path="outputs/visualizations"):
    metric_names = ['Precision', 'Recall', 'F1', 'IoU']
    metric_values = np.array([
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['iou']
    ]).reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(metric_values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', fontsize=11)
    
    # Add labels
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_yticks([0])
    ax.set_xticklabels(metric_names)
    ax.set_yticklabels(['Test Metrics'])
    ax.set_title('Evaluation Metrics', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(metric_names)):
        text = ax.text(i, 0, f'{metric_values[0, i]:.4f}',
                      ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'metrics_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


# -------------------------------
# Main
# -------------------------------
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ds = ChangeDetectionDataset(
        config['test']['pre_dir'],
        config['test']['post_dir'],
        config['test']['mask_dir'],
        transform=get_val_transforms()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config['test']['batch_size'],
        shuffle=False,
        num_workers=config['test']['num_workers']
    )

    # ✅ FIXED MODEL
    model = DualStreamAttentionUNet(
        eo_channels=config['model']['eo_channels'],
        sar_channels=config['model']['sar_channels'],
        out_channels=config['model']['out_channels']
    ).to(device)

    model.load_state_dict(torch.load('best_model.pth', map_location=device))

    metrics = evaluate(
        model,
        test_loader,
        device,
        save_vis=True,
        vis_dir="outputs/visualizations"
    )

    print("\n📊 Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n📌 Confusion Matrix:")
    print(f"TP: {metrics['tp']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"TN: {metrics['tn']}")
    
    # 📊 Generate visualizations
    vis_dir = "outputs/visualizations"
    plot_confusion_matrix(metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn'], vis_dir)
    plot_metrics_heatmap(metrics, vis_dir)
    
    print(f"\n✅ Heatmaps saved to {vis_dir}/")
    print("   - confusion_matrix.png")
    print("   - metrics_heatmap.png")


if __name__ == '__main__':
    main()