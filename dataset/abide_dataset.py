"""
PyTorch Dataset and DataLoader for ABIDE (Autism Brain Imaging Data Exchange) MRI Data.
Supports 2D slice directories and phenotypic CSV metadata annotations.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .mri_preprocessor import load_mri_file, get_mri_transforms


class ABIDEDataset(Dataset):
    """
    ABIDE MRI Dataset Loader.
    Expected folder layout (standard image classification setup):
      data/
        train/
          ASD/
            subject_001_slice_01.png ...
          TC/
            subject_101_slice_01.png ...
        val/
          ASD/
          TC/
        test/
          ASD/
          TC/
    """
    def __init__(self, root_dir, split="train", image_size=(224, 224), transform=None):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.image_size = image_size
        self.transform = transform or get_mri_transforms(image_size, is_train=(split == "train"))

        self.samples = []
        self.class_to_idx = {"TC": 0, "ASD": 1}

        # Search for ASD (Autism Spectrum Disorder) and TC (Typical Control) directories
        for class_name, label in self.class_to_idx.items():
            class_folder = self.root_dir / class_name
            if class_folder.exists():
                valid_extensions = {".png", ".jpg", ".jpeg", ".npy", ".nii", ".gz", ".dcm"}
                for file_path in class_folder.glob("*.*"):
                    if file_path.suffix.lower() in valid_extensions:
                        self.samples.append((str(file_path), label))

        if len(self.samples) == 0:
            print(f"[Warning] No MRI images found in {self.root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            # Load and normalize raw MRI slice/volume
            mri_data = load_mri_file(file_path)

            # If 3D, take center slice for 2D classification
            if mri_data.ndim == 3:
                mid = mri_data.shape[2] // 2
                mri_data = mri_data[:, :, mid]

            # Convert to uint8 for torchvision transformations
            uint8_img = (mri_data * 255.0).astype(np.uint8)
            tensor_img = self.transform(uint8_img)

            return tensor_img, torch.tensor(label, dtype=torch.long), file_path

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            dummy = torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
            return dummy, torch.tensor(label, dtype=torch.long), file_path


def get_data_loaders(data_dir, batch_size=16, image_size=(224, 224), num_workers=0):
    """
    Creates and returns DataLoaders for train, val, and test splits.
    """
    train_dataset = ABIDEDataset(data_dir, split="train", image_size=image_size)
    val_dataset = ABIDEDataset(data_dir, split="val", image_size=image_size)
    test_dataset = ABIDEDataset(data_dir, split="test", image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
