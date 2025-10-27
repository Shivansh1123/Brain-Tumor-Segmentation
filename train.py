"""
Training module for Brain Tumor Segmentation models
"""

import torch
from tqdm import tqdm
from models import EarlyStopping, dice_pytorch, iou_pytorch


def training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler, device, is_3d=False):
    """
    Training loop for both 2D and 3D models

    Args:
        epochs: Number of training epochs
        model: PyTorch model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        optimizer: Optimizer for model
        loss_fn: Loss function
        lr_scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        is_3d: Whether the model is 3D

    Returns:
        Dictionary with training history
    """
    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)

    for epoch in range(1, epochs + 1):
        running_loss = 0
        train_samples = 0
        model.train()

        for data in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            if is_3d:
                # Handle list format from 3D dataset
                images_list, masks_list = data
                for images, masks in zip(images_list, masks_list):
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

        lr_scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            early_stopping.load_weights(model)
            break

    model.eval()
    return history


def evaluate_model(model, test_loader, device, model_name, is_3d=False):
    """
    Evaluate model on test set

    Args:
        model: Trained model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        model_name: Name of model for logging
        is_3d: Whether the model is 3D

    Returns:
        Dictionary with evaluation metrics
    """
    import numpy as np

    model.eval()
    all_dice = []
    all_iou = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            if is_3d:
                images_list, masks_list = data
                for images, masks in zip(images_list, masks_list):
                    images = images.unsqueeze(0).to(device)
                    masks = masks.unsqueeze(0).to(device)

                    predictions = model(images)

                    pred_binary = (predictions > 0.5).float()

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

                pred_binary = (predictions > 0.5).float()

                dice = dice_pytorch(pred_binary, masks).cpu().numpy()
                iou = iou_pytorch(pred_binary, masks).cpu().numpy()

                all_dice.extend(dice.flatten())
                all_iou.extend(iou.flatten())

    return {
        'model': model_name,
        'dice': np.mean(all_dice),
        'dice_std': np.std(all_dice),
        'iou': np.mean(all_iou),
        'iou_std': np.std(all_iou),
    }
