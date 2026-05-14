# GalaxEye Binary Change Detection

## Project Title & Description

This repository implements a binary change detection pipeline for co-registered EO-SAR pre-event and post-event image pairs. The task is to predict pixel-level change masks for disaster-affected scenes.

**Approach**: We use a Dual-Stream Attention U-Net architecture that processes EO (RGB) and SAR (grayscale) images separately through dedicated encoder streams, then fuses them using channel (SE) and spatial attention mechanisms. The model employs focal loss and dice loss (0.5/0.5 weighting) for better handling of class imbalance, with morphological post-processing to reduce noise.

**Key Features**:
- Dual-stream encoders for EO and SAR modalities
- Attention-based fusion with SE and spatial attention blocks
- Balanced focal-dice loss for imbalanced segmentation
- Probability thresholding (0.5) + morphological opening
- Data augmentation: random crop (384×384), flip, rotation, noise

## Requirements

- Python 3.12
- torch==2.3.1+cu118
- torchvision==0.18.1+cu118
- rasterio==1.5.0
- numpy==2.4.4
- matplotlib==3.10.9
- scikit-learn==1.8.0
- tqdm==4.67.3
- albumentations==2.0.8
- opencv-python==4.13.0.92

## Environment Setup

```bash
# Create virtual environment
python3 -m venv galaxeye_env

# Activate environment
source galaxeye_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Expected directory layout after placing the dataset:

```
GalaxEye/
├── train/
│   ├── pre-event/     # EO images (pre-disaster)
│   ├── post-event/    # SAR images (post-disaster)
│   └── target/        # Ground truth change masks
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── test/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── config.yaml        # Configuration file
├── requirements.txt   # Dependencies
└── [model files]
```

## Training

Train the model from scratch using the configuration file:

```bash
python train.py
```

This will:
- Load configuration from `config.yaml`
- Train for 80 epochs with early stopping
- Use ReduceLROnPlateau scheduler
- Save best model as `best_model.pth`
- Generate training curves as `train_val_curves.png`

### Training Outputs
- `best_model.pth` - Best model weights (saved when validation IoU improves)
- `train_val_curves.png` - Training and validation loss/F1 curves

## Evaluation

Evaluate the trained model on test data:

```bash
python eval.py
```

This will:
- Load model weights from `best_model.pth`
- Evaluate on test split defined in `config.yaml`
- Generate prediction visualizations in `outputs/visualizations/`
- Print comprehensive metrics and confusion matrix
- Save confusion matrix and metrics heatmaps

### Output Files
After evaluation, the following files are generated:
- `outputs/visualizations/confusion_matrix.png` - Confusion matrix heatmap
- `outputs/visualizations/metrics_heatmap.png` - Performance metrics visualization
- `outputs/visualizations/sample_*.png` - Prediction visualizations (first 10 samples)

## Model Weights

**Local**: `best_model.pth` (generated after training)

**Download Link**: [Upload best_model.pth to Google Drive/HuggingFace and add link here]

*Note: The trained model weights will be available for download after uploading to a cloud storage service.*

## Results

### Validation Results
| Metric | Value |
|--------|-------|
| Precision | 0.6434 |
| Recall | 0.4470 |
| F1 Score | 0.5275 |
| IoU | 0.3582 |

### Test Results
| Metric | Value |
|--------|-------|
| Precision | 0.5737 |
| Recall | 0.6246 |
| F1 Score | 0.5981 |
| IoU | 0.4266 |

### Confusion Matrix (Test Set)
- True Positives: 3,256,227
- False Positives: 2,419,405
- False Negatives: 1,957,285
- True Negatives: 73,107,435

## Citation / References

### Papers
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 (pp. 234-241).

- Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (pp. 2980-2988).

### Codebases
- PyTorch Segmentation Models: https://github.com/qubvel/segmentation_models.pytorch
- Albumentations: https://github.com/albumentations-team/albumentations

### Datasets
- No external datasets were used. The model was trained on proprietary EO-SAR change detection data.
