"""
Model definitions for Brain Tumor Segmentation
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.networks.nets import UNet, UNETR


class EarlyStopping:
    """Stops training when loss stops decreasing in a PyTorch module."""

    def __init__(self, patience: int = 6, min_delta: float = 0, weights_path: str = 'weights.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.weights_path = weights_path

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_weights(self, model: nn.Module):
        return model.load_state_dict(torch.load(self.weights_path))


def get_unet_2d(device):
    """Create 2D U-Net model"""
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    model.to(device)
    return model


def get_unet_3d(device):
    """Create 3D U-Net model"""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    model.to(device)
    return model


def get_unetr_3d(device, img_size=(40, 256, 256)):
    """Create UNETR model (Transformer-based 3D)"""
    model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=img_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="conv",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        spatial_dims=3,
    )
    model.to(device)
    return model


def get_nnunet_like_3d(device):
    """Create nnU-Net-like model (deeper U-Net with dropout)"""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256),  # Deeper than standard
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.2,
    )
    model.to(device)
    return model


def create_model_optimizer_scheduler(model, learning_rate=0.001, weight_decay=0.0):
    """Create optimizer and learning rate scheduler for a model"""
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=3, factor=0.5)
    return optimizer, scheduler


def get_loss_function():
    """Get loss function for segmentation"""
    return nn.BCEWithLogitsLoss()


def iou_pytorch(predictions, labels, e=1e-7):
    """Calculates Intersection over Union for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()

    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))

    iou = (intersection + e) / (union + e)
    return iou


def dice_pytorch(predictions, labels, e=1e-7):
    """Calculates Dice coefficient for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()

    intersection = (predictions & labels).float().sum((1, 2))
    return ((2 * intersection) + e) / (predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e)


def bce_dice_loss(output, target, alpha=0.01):
    """Combined BCE and Dice loss"""
    bce = nn.functional.binary_cross_entropy(output, target)
    soft_dice = 1 - dice_pytorch(output, target).mean()
    return bce + alpha * soft_dice


class ModelConfig:
    """Configuration for different models"""

    configs = {
        'unet_2d': {
            'name': '2D U-Net (MONAI)',
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'is_3d': False,
        },
        'unet_3d': {
            'name': '3D U-Net (MONAI)',
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'is_3d': True,
        },
        'unetr_3d': {
            'name': 'UNETR (Transformer-based 3D)',
            'learning_rate': 0.0005,
            'weight_decay': 0.0,
            'is_3d': True,
        },
        'nnunet_like_3d': {
            'name': 'nnU-Net-like (Deep 3D U-Net)',
            'learning_rate': 0.001,
            'weight_decay': 3e-5,
            'is_3d': True,
        },
    }

    @staticmethod
    def get_config(model_key):
        return ModelConfig.configs.get(model_key, {})
