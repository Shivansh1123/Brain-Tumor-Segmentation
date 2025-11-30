"""
Data loading and preprocessing for medical image segmentation
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import albumentations as A
from sklearn.model_selection import train_test_split


def get_file_row(path):
    """Produces ID of a patient, image and mask filenames from a particular path"""
    path_no_ext, ext = os.path.splitext(path)
    filename = os.path.basename(path)

    patient_id = '_'.join(filename.split('_')[:3])

    return [patient_id, path, f'{path_no_ext}_mask{ext}']


def load_data(data_dir):
    """Load file paths and create DataFrame"""
    file_paths = glob(f'{data_dir}/*/*[0-9].tif')
    print(f"Total files found: {len(file_paths)}")

    filenames_df = pd.DataFrame(
        (get_file_row(filename) for filename in file_paths),
        columns=['Patient', 'image_filename', 'mask_filename']
    )
    print(f"Total patient records: {len(filenames_df)}")

    return filenames_df


def filter_zero_masks(df):
    """Remove samples that have no tumor (all-zero masks)"""
    print(f"\nFiltering out samples with zero masks...")
    print(f"Original dataset size: {len(df)}")

    valid_indices = []
    for idx, row in df.iterrows():
        mask_path = row['mask_filename']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Keep only if mask has tumor pixels
        if mask is not None and np.count_nonzero(mask) > 0:
            valid_indices.append(idx)

    filtered_df = df.loc[valid_indices].reset_index(drop=True)
    removed = len(df) - len(filtered_df)

    print(f"Filtered dataset size: {len(filtered_df)}")
    print(f"Removed {removed} samples with zero masks ({100*removed/len(df):.1f}%)")

    return filtered_df


def split_data(df, test_size=0.3, valid_size=0.5, random_state=42):
    """Split data into train, validation, and test sets"""
    # Filter out zero-mask samples first
    df = filter_zero_masks(df)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    test_df, valid_df = train_test_split(test_df, test_size=valid_size, random_state=random_state)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, valid_df, test_df


class MriDataset(Dataset):
    """Dataset class for 2D MRI slices"""

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
        mask = mask.astype(np.float32) / 255.0  # Normalize mask to [0, 1]
        mask = torch.from_numpy(mask)
        return img, mask


# class MriDataset3D(Dataset):
#     """Dataset class for 3D MRI volumes"""

#     def __init__(self, df, transform=None, max_depth=40):
#         super(MriDataset3D, self).__init__()
#         self.df = df.reset_index(drop=True)
#         self.transform = transform
#         self.max_depth = max_depth
#         self.patients = self.df.groupby('Patient').groups
#         self.patient_ids = list(self.patients.keys())

#     def __len__(self):
#         return len(self.patient_ids)

#     def __getitem__(self, idx):
#         patient_id = self.patient_ids[idx]
#         indices = list(self.patients[patient_id])

#         # Load all slices for a patient
#         images = []
#         masks = []
#         for i in indices:
#             row = self.df.loc[i]
#             img = cv2.imread(row['image_filename'], cv2.IMREAD_GRAYSCALE)
#             mask = cv2.imread(row['mask_filename'], cv2.IMREAD_GRAYSCALE)

#             if img is None or mask is None:
#                 continue

#             images.append(img)
#             masks.append(mask // 255)

#         if len(images) == 0:
#             raise ValueError(f"No valid images found for patient {patient_id}")

#         # Stack to create 3D volume (D, H, W)
#         images = np.stack(images, axis=0)
#         masks = np.stack(masks, axis=0)

#         # Pad or crop to fixed depth (max_depth)
#         current_depth = images.shape[0]

#         if current_depth < self.max_depth:
#             # Pad with zeros at the end
#             pad_size = self.max_depth - current_depth
#             images = np.pad(images, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
#             masks = np.pad(masks, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
#         elif current_depth > self.max_depth:
#             # Crop to max_depth (take middle slices)
#             start = (current_depth - self.max_depth) // 2
#             images = images[start:start + self.max_depth]
#             masks = masks[start:start + self.max_depth]

#         # Convert to (C, D, H, W) format for MONAI
#         images = torch.Tensor(images).unsqueeze(0).float()
#         masks = torch.Tensor(masks).unsqueeze(0).float()

#         return images, masks


def create_dataloaders(train_df, valid_df, test_df, batch_size_2d=16, use_3d=False):
    """Create data loaders for 2D datasets"""

    # 2D augmentation - Proven working configuration
    transform = A.Compose([
        # Geometric transforms
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Affine(scale=(0.9, 1.1), shear=(-10, 10), p=0.3),

        # Intensity transforms
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),

        # Morphological transforms
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
    ], bbox_params=None)

    # 2D datasets
    train_dataset_2d = MriDataset(train_df, transform)
    valid_dataset_2d = MriDataset(valid_df)
    test_dataset_2d = MriDataset(test_df)

    train_loader_2d = DataLoader(train_dataset_2d, batch_size=batch_size_2d, shuffle=True)
    valid_loader_2d = DataLoader(valid_dataset_2d, batch_size=batch_size_2d, shuffle=False)
    test_loader_2d = DataLoader(test_dataset_2d, batch_size=1)

    dataloaders = {
        'train_2d': train_loader_2d,
        'valid_2d': valid_loader_2d,
        'test_2d': test_loader_2d,
    }

    print("\n2D Dataloaders created successfully!")

    return dataloaders
