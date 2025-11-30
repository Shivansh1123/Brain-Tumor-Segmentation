"""
Utility functions for medical image segmentation
"""

import torch
import numpy as np
from tqdm import tqdm


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 6, min_delta: float = 0, weights_path: str = 'weights.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.weights_path = weights_path

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_weights(self, model: torch.nn.Module):
        return model.load_state_dict(torch.load(self.weights_path))


def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculate Intersection over Union (IoU) metric for both 2D and 3D"""
    predictions = torch.where(predictions > 0.5, 1, 0).float()
    labels = labels.float()

    # Flatten all spatial dimensions
    predictions_flat = predictions.view(predictions.shape[0], -1)
    labels_flat = labels.view(labels.shape[0], -1)

    intersection = (predictions_flat * labels_flat).sum(dim=1)
    union = (predictions_flat + labels_flat - predictions_flat * labels_flat).sum(dim=1)

    iou = (intersection + e) / (union + e)
    return iou


def dice_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculate Dice coefficient metric for both 2D and 3D"""
    predictions = torch.where(predictions > 0.5, 1, 0).float()
    labels = labels.float()

    # Flatten all spatial dimensions
    predictions_flat = predictions.view(predictions.shape[0], -1)
    labels_flat = labels.view(labels.shape[0], -1)

    intersection = (predictions_flat * labels_flat).sum(dim=1)
    pred_sum = predictions_flat.sum(dim=1)
    label_sum = labels_flat.sum(dim=1)

    dice = ((2 * intersection) + e) / (pred_sum + label_sum + e)
    return dice


def BCE_dice(output, target, alpha=0.01):
    """Combined Binary Cross Entropy and Dice loss"""
    bce = torch.nn.functional.binary_cross_entropy(output, target)
    soft_dice = 1 - dice_pytorch(output, target).mean()
    return bce + alpha * soft_dice


def save_epoch_visualization(model, valid_loader, device, epoch, viz_dir='viz', num_samples=6):
    """Save predictions visualization for a specific epoch"""
    import os
    import matplotlib.pyplot as plt

    os.makedirs(viz_dir, exist_ok=True)

    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    count = 0
    with torch.no_grad():
        for images, masks in valid_loader:
            if count >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            if len(predictions.shape) == 4 and predictions.shape[1] == 1:
                predictions = predictions.squeeze(1)

            # Convert logits to probabilities
            pred_probs = torch.sigmoid(predictions)
            pred_binary = (pred_probs > 0.5).float()

            # Plot
            for b in range(min(images.shape[0], num_samples - count)):
                if count >= num_samples:
                    break

                img = images[b, 0].cpu().numpy()
                mask = masks[b].cpu().numpy()
                pred = pred_binary[b].cpu().numpy()

                axes[count, 0].imshow(img, cmap='gray')
                axes[count, 0].set_title(f'Image')
                axes[count, 0].axis('off')

                axes[count, 1].imshow(img, cmap='gray')
                axes[count, 1].imshow(mask, cmap='Reds', alpha=0.5)
                axes[count, 1].set_title(f'Ground Truth')
                axes[count, 1].axis('off')

                axes[count, 2].imshow(img, cmap='gray')
                axes[count, 2].imshow(pred, cmap='Reds', alpha=0.5)
                axes[count, 2].set_title(f'Prediction')
                axes[count, 2].axis('off')

                count += 1

    plt.suptitle(f'Epoch {epoch} - Predictions vs Ground Truth', fontsize=14)
    plt.tight_layout()

    plot_path = os.path.join(viz_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Visualization saved: {plot_path}")


def training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler, device, is_3d=False, viz_dir='viz'):
    """Main training loop for the model"""
    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)

    for epoch in range(1, epochs + 1):
        running_loss = 0
        train_samples = 0
        model.train()

        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            if is_3d:
                # Handle list format from custom collate function
                images_list, masks_list = data
                for images, masks in zip(images_list, masks_list):
                    # images and masks are already (C, D, H, W), add batch dimension
                    images = images.unsqueeze(0).to(device)  # (1, C, D, H, W)
                    masks = masks.unsqueeze(0).to(device)

                    predictions = model(images)

                    loss = loss_fn(predictions, masks)
                    running_loss += loss.item()
                    train_samples += 1

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                images, masks = data
                images, masks = images.to(device), masks.to(device)

                predictions = model(images)
                if len(predictions.shape) == 4:
                    predictions = predictions.squeeze(1)

                loss = loss_fn(predictions, masks)
                running_loss += loss.item() * images.size(0)
                train_samples += images.size(0)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_valid_loss = 0
            valid_samples = 0

            for data in valid_loader:
                if is_3d:
                    images_list, masks_list = data
                    for images, masks in zip(images_list, masks_list):
                        images = images.unsqueeze(0).to(device)
                        masks = masks.unsqueeze(0).to(device)

                        predictions = model(images)

                        running_dice += dice_pytorch(predictions, masks).sum().item()
                        running_IoU += iou_pytorch(predictions, masks).sum().item()
                        loss = loss_fn(predictions, masks)
                        running_valid_loss += loss.item()
                        valid_samples += 1
                else:
                    images, masks = data
                    images, masks = images.to(device), masks.to(device)

                    predictions = model(images)
                    if len(predictions.shape) == 4:
                        predictions = predictions.squeeze(1)

                    running_dice += dice_pytorch(predictions, masks).sum().item()
                    running_IoU += iou_pytorch(predictions, masks).sum().item()
                    loss = loss_fn(predictions, masks)
                    running_valid_loss += loss.item() * images.size(0)
                    valid_samples += images.size(0)

        train_loss = running_loss / max(1, train_samples)
        val_loss = running_valid_loss / max(1, valid_samples)
        val_dice = running_dice / max(1, valid_samples)
        val_IoU = running_IoU / max(1, valid_samples)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_IoU'].append(val_IoU)
        history['val_dice'].append(val_dice)

        print(f'Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.6f} | Val IoU: {val_IoU:.6f}')

        # Save visualization every 10 epochs
        if epoch % 10 == 0:
            save_epoch_visualization(model, valid_loader, device, epoch, viz_dir=viz_dir, num_samples=6)

        lr_scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            early_stopping.load_weights(model)
            break

    model.eval()
    return history


def evaluate_model(model, test_loader, device, model_name, is_3d=False):
    """Evaluate model on test dataset"""
    model.eval()
    all_dice = []
    all_iou = []
    pred_means = []  # Track mean prediction values
    mask_means = []  # Track mean mask values

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            if is_3d:
                images_list, masks_list = data
                for images, masks in zip(images_list, masks_list):
                    images = images.unsqueeze(0).to(device)
                    masks = masks.unsqueeze(0).to(device)

                    predictions = model(images)

                    # Convert logits to probabilities and threshold
                    pred_probs = torch.sigmoid(predictions)
                    pred_binary = (pred_probs > 0.5).float()

                    dice = dice_pytorch(pred_binary, masks).cpu().numpy()
                    iou = iou_pytorch(pred_binary, masks).cpu().numpy()

                    all_dice.extend(dice.flatten())
                    all_iou.extend(iou.flatten())
            else:
                images, masks = data
                images = images.to(device)
                masks = masks.to(device)

                predictions = model(images)

                if predictions.shape != masks.shape:
                    predictions = predictions.squeeze(1)

                # Convert logits to probabilities and threshold
                pred_probs = torch.sigmoid(predictions)
                pred_binary = (pred_probs > 0.5).float()

                dice = dice_pytorch(pred_binary, masks).cpu().numpy()
                iou = iou_pytorch(pred_binary, masks).cpu().numpy()

                all_dice.extend(dice.flatten())
                all_iou.extend(iou.flatten())

                # Track prediction statistics
                pred_means.append(predictions.cpu().mean().item())
                mask_means.append(masks.cpu().mean().item())

    # Print diagnostic info with enhanced details
    print(f"\nDiagnostic Info:")
    print(f"  Mean prediction value: {np.mean(pred_means):.6f} (should vary)")
    print(f"  Mean mask value: {np.mean(mask_means):.6f} (ground truth)")
    print(f"  Num samples with non-zero predictions: {sum(1 for p in pred_means if p > 0.1)}")

    # Additional diagnostics for debugging
    if len(pred_means) > 0:
        print(f"  Prediction range: [{min(pred_means):.6f}, {max(pred_means):.6f}]")
        print(f"  Std of predictions: {np.std(pred_means):.6f} (variability)")

    # Check if model is collapse (all zeros)
    if len(all_dice) > 0 and np.mean(all_dice) < 0.01:
        print(f"\n⚠️  WARNING: Model predictions are near zero!")
        print(f"  Dice: {np.mean(all_dice):.6f} (should be > 0.3 for learning)")
        print(f"  This indicates model collapse. Consider:")
        print(f"    - Increasing pos_weight (currently 5.0 for nnU-Net)")
        print(f"    - Reducing learning rate")
        print(f"    - Using smaller model or more data")

    return {
        'model': model_name,
        'dice': np.mean(all_dice),
        'dice_std': np.std(all_dice),
        'iou': np.mean(all_iou),
        'iou_std': np.std(all_iou),
    }
