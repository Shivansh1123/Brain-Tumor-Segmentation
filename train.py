"""
Main training script for medical image segmentation
"""

import torch
import argparse
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.networks.nets import UNet, UNETR

from data import load_data, split_data, create_dataloaders
from utils import training_loop, evaluate_model


def create_model(model_type, device='cpu'):
    """Create and return the specified model"""

    if model_type == 'unet_2d':
        print("=" * 60)
        print("CREATING MODEL: Simple 2D U-Net (Winning Config)")
        print("=" * 60)
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(8, 16, 32),  # Simple: 3 levels (proven to work well)
            strides=(2, 2),        # 2 downsampling steps
            num_res_units=0,       # No residuals (simplicity is key)
        )

    elif model_type == 'unetr_2d':
        print("=" * 60)
        print("CREATING MODEL: UNETR (Transformer-based, 2D)")
        print("=" * 60)
        model = UNETR(
            in_channels=1,
            out_channels=1,
            img_size=(256, 256),
            feature_size=8,
            hidden_size=256,
            mlp_dim=1024,
            num_heads=4,
            norm_name="instance",
            conv_block=True,
            res_block=True,
            spatial_dims=2,
            dropout_rate=0.2,
        )

    elif model_type == 'nnunet_2d':
        print("=" * 60)
        print("CREATING MODEL: nnU-Net (2D) - Fixed Configuration")
        print("=" * 60)
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(8, 16, 32, 64),  # Reduced from (16,32,64,128) - was too deep
            strides=(2, 2, 2),         # 3 downsampling steps
            num_res_units=2,           # Residual connections
            dropout=0.3,               # Increased from 0.2 - more regularization
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    return model


def visualize_predictions(model, test_loader, device, model_name, results_dir, is_3d=False, num_samples=6):
    """Visualize predictions vs ground truth and save plots"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    count = 0
    with torch.no_grad():
        for data in test_loader:
            if count >= num_samples:
                break

            try:
                if is_3d:
                    images_list, masks_list = data
                    for images, masks in zip(images_list, masks_list):
                        if count >= num_samples:
                            break

                        images = images.unsqueeze(0).to(device)  # (1, C, D, H, W)
                        masks = masks.unsqueeze(0).to(device)

                        predictions = model(images)
                        pred_binary = (predictions > 0.5).float()

                        # Get middle slice for visualization
                        mid_slice = images.shape[2] // 2
                        img_slice = images[0, 0, mid_slice].cpu().numpy()
                        mask_slice = masks[0, 0, mid_slice].cpu().numpy()
                        pred_slice = pred_binary[0, 0, mid_slice].cpu().numpy()

                        # Plot
                        axes[count, 0].imshow(img_slice, cmap='gray')
                        axes[count, 0].set_title(f'Image {count+1}')
                        axes[count, 0].axis('off')

                        axes[count, 1].imshow(img_slice, cmap='gray')
                        axes[count, 1].imshow(mask_slice, cmap='Reds', alpha=0.5)
                        axes[count, 1].set_title(f'Ground Truth {count+1}')
                        axes[count, 1].axis('off')

                        axes[count, 2].imshow(img_slice, cmap='gray')
                        axes[count, 2].imshow(pred_slice, cmap='Reds', alpha=0.5)
                        axes[count, 2].set_title(f'Prediction {count+1}')
                        axes[count, 2].axis('off')

                        count += 1
                else:
                    images, masks = data
                    images = images.to(device)  # (B, 1, H, W)
                    masks = masks.to(device)

                    predictions = model(images)
                    if len(predictions.shape) == 4 and predictions.shape[1] == 1:
                        predictions = predictions.squeeze(1)

                    # Convert logits to probabilities and threshold
                    pred_probs = torch.sigmoid(predictions)
                    pred_binary = (pred_probs > 0.5).float()

                    # Process each image in batch
                    for b in range(images.shape[0]):
                        if count >= num_samples:
                            break

                        img_slice = images[b, 0].cpu().numpy()
                        mask_slice = masks[b].cpu().numpy()
                        pred_slice = pred_binary[b].cpu().numpy()

                        # Plot
                        axes[count, 0].imshow(img_slice, cmap='gray')
                        axes[count, 0].set_title(f'Image {count+1}')
                        axes[count, 0].axis('off')

                        axes[count, 1].imshow(img_slice, cmap='gray')
                        axes[count, 1].imshow(mask_slice, cmap='Reds', alpha=0.5)
                        axes[count, 1].set_title(f'Ground Truth {count+1}')
                        axes[count, 1].axis('off')

                        axes[count, 2].imshow(img_slice, cmap='gray')
                        axes[count, 2].imshow(pred_slice, cmap='Reds', alpha=0.5)
                        axes[count, 2].set_title(f'Prediction {count+1}')
                        axes[count, 2].axis('off')

                        count += 1
            except Exception as e:
                print(f"Warning: Error processing sample {count}: {e}")
                continue

    plt.suptitle(f'{model_name} - Predictions vs Ground Truth', fontsize=16, y=0.995)
    plt.tight_layout()

    # Save figure
    plot_filename = os.path.join(results_dir, f"{model_name}_predictions.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Prediction plot saved to {plot_filename}")
    plt.close()


def train_model(config):
    """Main training function"""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    filenames_df = load_data(config['data_dir'])
    train_df, valid_df, test_df = split_data(filenames_df)

    # Create dataloaders (2D only)
    dataloaders = create_dataloaders(
        train_df, valid_df, test_df,
        batch_size_2d=config.get('batch_size_2d', 16),
        use_3d=False
    )

    # Create model
    model = create_model(config['model_type'], device=device)

    # Setup training - all 2D models
    train_loader = dataloaders['train_2d']
    valid_loader = dataloaders['valid_2d']
    test_loader = dataloaders['test_2d']

    # Model-specific configurations
    if config['model_type'] == 'nnunet_2d':
        # nnU-Net needs more aggressive pos_weight to prevent collapse
        pos_weight = torch.tensor([5.0]).to(device)
        learning_rate = 0.0005  # Reduced LR for deeper model stability
        lr_patience = 4         # Earlier LR reduction
    else:
        # U-Net and UNETR use proven configuration
        pos_weight = torch.tensor([3.0]).to(device)
        learning_rate = 0.001
        lr_patience = 5

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Learning rate scheduler with model-specific patience
    lr_scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=lr_patience,
        factor=0.5,
        min_lr=1e-6
    )

    # Train model
    print("\n" + "=" * 60)
    print(f"TRAINING {config['model_type'].upper()}")
    print("=" * 60)

    # Create viz directory
    import os
    viz_dir = 'viz'
    os.makedirs(viz_dir, exist_ok=True)

    history = training_loop(
        epochs=config['epochs'],
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        device=device,
        is_3d=False,
        viz_dir=viz_dir
    )

    print(f"\n✓ {config['model_type'].upper()} Training Complete!")

    # Evaluate model
    print("\n" + "=" * 60)
    print(f"EVALUATING {config['model_type'].upper()}")
    print("=" * 60)
    results = evaluate_model(model, test_loader, device, config['model_type'], is_3d=False)

    print(f"\nTest Results for {config['model_type']}:")
    print(f"  Dice: {results['dice']:.6f} ± {results['dice_std']:.6f}")
    print(f"  IoU:  {results['iou']:.6f} ± {results['iou_std']:.6f}")

    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Generate and save prediction visualizations
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, device, config['model_type'], results_dir, is_3d=False, num_samples=6)

    # Save results in TXT format
    results['history'] = history
    txt_filename = os.path.join(results_dir, f"{config['model_type']}_results.txt")
    with open(txt_filename, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Model Results: {config['model_type'].upper()}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
        f.write(f"Epochs: {config['epochs']}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"TEST METRICS\n")
        f.write(f"{'='*60}\n")
        f.write(f"Dice Coefficient: {results['dice']:.6f} ± {results['dice_std']:.6f}\n")
        f.write(f"IoU (Jaccard):    {results['iou']:.6f} ± {results['iou_std']:.6f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"TRAINING HISTORY\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Val Dice':<15} {'Val IoU':<15}\n")
        f.write(f"{'-'*60}\n")
        for epoch, (tl, vl, vd, vi) in enumerate(zip(
            history['train_loss'],
            history['val_loss'],
            history['val_dice'],
            history['val_IoU']
        ), 1):
            f.write(f"{epoch:<8} {tl:<15.6f} {vl:<15.6f} {vd:<15.6f} {vi:<15.6f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Best Validation Dice: {max(history['val_dice']):.6f}\n")
        f.write(f"Best Validation IoU: {max(history['val_IoU']):.6f}\n")

    # Save JSON results as well
    json_filename = os.path.join(results_dir, f"{config['model_type']}_results.json")
    with open(json_filename, 'w') as f:
        json_results = {
            'model': results['model'],
            'dice': float(results['dice']),
            'dice_std': float(results['dice_std']),
            'iou': float(results['iou']),
            'iou_std': float(results['iou_std']),
            'history': {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'val_IoU': history['val_IoU'],
                'val_dice': history['val_dice']
            }
        }
        json.dump(json_results, f, indent=2)

    # Save model
    model_filename = os.path.join(results_dir, f"{config['model_type']}_model.pt")
    torch.save(model.state_dict(), model_filename)
    print(f"\nModel saved to {model_filename}")
    print(f"Results saved to {txt_filename}")
    print(f"JSON saved to {json_filename}")

    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train medical image segmentation models')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--model-type', type=str, default='unet_2d',
                        choices=['unet_2d', 'unetr_2d', 'nnunet_2d'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')

    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size_2d': args.batch_size,
    }

    model, results = train_model(config)


if __name__ == '__main__':
    main()
