# Brain Tumor Segmentation with Deep Learning

A comprehensive deep learning solution for automated brain tumor segmentation from MRI scans using state-of-the-art medical image analysis techniques.

## Overview

This project implements and compares multiple advanced neural network architectures for 3D volumetric medical image segmentation:

- **U-Net** - Baseline slice-by-slice segmentation
- **UNETR** - Transformer-based architecture for medical imaging
- **nnU-Net** - Self-configuring framework optimized for medical image analysis

## Dataset

[LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) - 110 patients with low-grade glioma brain tumors

- Variable MRI slice counts (25-40 slices per patient)
- Normalized to fixed 40-slice volumes for batch processing
- Grayscale MRI images (256×256 pixels)
- Binary segmentation masks

## Key Features

✓ **Robust preprocessing pipeline** - Handles variable-size patient MRI scans with padding/cropping
✓ **4 model comparison** - Side-by-side evaluation of different architectures
✓ **3D volume support** - True 3D convolutions for volumetric medical imaging
✓ **Visualization** - Ground truth vs predicted segmentation overlays
✓ **nnU-Net integration** - Self-configuring framework with optimal hyperparameters

## Requirements

```bash
pip install -q segmentation-models-pytorch monai nnunet
pip install opencv-python numpy pandas matplotlib albumentations scikit-learn tqdm
```

## Usage

1. **Prepare data** - Download LGG MRI dataset from Kaggle
2. **Run notebook** - Execute all cells to train and compare models
3. **Evaluate** - View performance metrics and visualizations
4. **nnU-Net training** - Use CLI commands for optimal results:
   ```bash
   nnUNetv2_plan_and_preprocess_task -t 1
   nnUNetv2_train 1 3d_fullres 0 -device cuda
   ```

## Architecture Details

All models use:
- Input: 3D MRI volumes (40×256×256)
- Output: Binary segmentation mask (tumor/non-tumor)
- Loss: Binary Cross-Entropy with Dice Loss
- Optimizer: Adam with learning rate scheduling

## Project Highlights

- Engineered robust preprocessing pipeline with 3D volume padding/cropping
- Implemented and compared 3 advanced neural network architectures
- Conducted comparative study across different segmentation approaches
- Built comprehensive evaluation framework with visualization tools

---

**Dataset Source:** [Kaggle LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
