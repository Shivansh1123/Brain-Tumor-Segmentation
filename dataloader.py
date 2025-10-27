"""
Data loading and preprocessing module for Brain Tumor Segmentation
"""

import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


def get_file_row(path):
    """Produces ID of a patient, image and mask filenames from a particular path"""
    path_no_ext, ext = os.path.splitext(path)
    filename = os.path.basename(path)

    patient_id = '_'.join(filename.split('_')[:3])

    return [patient_id, path, f'{path_no_ext}_mask{ext}']


def load_dataset(files_dir):
    """Load and prepare dataset from directory"""
    file_paths = glob(f'{files_dir}/*/*[0-9].tif')
    print(f"Total files found: {len(file_paths)}")

    filenames_df = pd.DataFrame(
        (get_file_row(filename) for filename in file_paths),
        columns=['Patient', 'image_filename', 'mask_filename']
    )
    print(f"Total patient records: {len(filenames_df)}")
    return filenames_df


def split_dataset(df, test_size=0.3, valid_size=0.5, random_state=42):
    """Split dataset into train, validation, and test sets"""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    test_df, valid_df = train_test_split(test_df, test_size=valid_size, random_state=random_state)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, valid_df, test_df


class MriDataset(Dataset):
    """2D Dataset for slice-by-slice segmentation"""

    def __init__(self, df, transform=None):
        super(MriDataset, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, raw=False):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_filename'], cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        mask = cv2.imread(row['mask_filename'], cv2.IMREAD_GRAYSCALE)

        if raw:
            return img, mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        img = T.functional.to_tensor(img)  # Converts (H, W) to (1, H, W)
        mask = mask // 255
        mask = torch.Tensor(mask)
        return img, mask


class MriDataset3D(Dataset):
    """3D Dataset for volumetric segmentation"""

    def __init__(self, df, transform=None, max_depth=40):
        super(MriDataset3D, self).__init__()
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.max_depth = max_depth
        self.patients = self.df.groupby('Patient').groups
        self.patient_ids = list(self.patients.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        indices = list(self.patients[patient_id])

        # Load all slices for a patient
        images = []
        masks = []
        for i in indices:
            row = self.df.loc[i]
            img = cv2.imread(row['image_filename'], cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(row['mask_filename'], cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            images.append(img)
            masks.append(mask // 255)

        if len(images) == 0:
            raise ValueError(f"No valid images found for patient {patient_id}")

        # Stack to create 3D volume (D, H, W)
        images = np.stack(images, axis=0)
        masks = np.stack(masks, axis=0)

        # Pad or crop to fixed depth (max_depth)
        current_depth = images.shape[0]

        if current_depth < self.max_depth:
            # Pad with zeros at the end
            pad_size = self.max_depth - current_depth
            images = np.pad(images, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
            masks = np.pad(masks, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif current_depth > self.max_depth:
            # Crop to max_depth (take middle slices)
            start = (current_depth - self.max_depth) // 2
            images = images[start:start+self.max_depth]
            masks = masks[start:start+self.max_depth]

        # Convert to (C, D, H, W) format for MONAI
        images = torch.Tensor(images).unsqueeze(0).float()
        masks = torch.Tensor(masks).unsqueeze(0).float()

        return images, masks


def get_transforms():
    """Get augmentation transforms for 2D training"""
    return A.Compose([
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Rotate(limit=10, p=0.3),
    ])


def create_dataloaders(train_df, valid_df, test_df, batch_size_2d=16, batch_size_3d=2, max_depth=40):
    """Create 2D and 3D dataloaders"""

    # 2D transforms
    transform = get_transforms()

    # 2D datasets for 2D U-Net
    train_dataset_2d = MriDataset(train_df, transform)
    valid_dataset_2d = MriDataset(valid_df)
    test_dataset_2d = MriDataset(test_df)

    train_loader_2d = DataLoader(train_dataset_2d, batch_size=batch_size_2d, shuffle=True)
    valid_loader_2d = DataLoader(valid_dataset_2d, batch_size=batch_size_2d, shuffle=False)
    test_loader_2d = DataLoader(test_dataset_2d, batch_size=1)

    # 3D datasets with padding to fixed size
    train_dataset_3d = MriDataset3D(train_df, max_depth=max_depth)
    valid_dataset_3d = MriDataset3D(valid_df, max_depth=max_depth)
    test_dataset_3d = MriDataset3D(test_df, max_depth=max_depth)

    train_loader_3d = DataLoader(train_dataset_3d, batch_size=batch_size_3d, shuffle=True)
    valid_loader_3d = DataLoader(valid_dataset_3d, batch_size=batch_size_3d, shuffle=False)
    test_loader_3d = DataLoader(test_dataset_3d, batch_size=1)

    print("2D Dataloaders created successfully!")
    print("3D Dataloaders created successfully (padded to 40 slices)!")

    return {
        '2d': {'train': train_loader_2d, 'valid': valid_loader_2d, 'test': test_loader_2d},
        '3d': {'train': train_loader_3d, 'valid': valid_loader_3d, 'test': test_loader_3d}
    }
