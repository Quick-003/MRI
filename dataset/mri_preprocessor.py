"""
MRI Preprocessing and Transformation Pipeline for Brain Scans.
Supports NIfTI (.nii, .nii.gz), DICOM (.dcm), and 2D MRI slice formats (.png, .jpg, .npy).
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

import torch
import torchvision.transforms as T


def normalize_intensity(volume, method="minmax"):
    """
    Normalizes MRI voxel intensities.
    Methods:
      - 'minmax': scales values to [0, 1]
      - 'zscore': zero mean and unit variance (background excluded)
    """
    volume = volume.astype(np.float32)
    if method == "minmax":
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min > 1e-6:
            return (volume - v_min) / (v_max - v_min)
        return np.zeros_like(volume)
    elif method == "zscore":
        mask = volume > volume.mean() * 0.1  # crude foreground mask
        if mask.sum() > 0:
            mean = volume[mask].mean()
            std = volume[mask].std()
            if std > 1e-6:
                normed = (volume - mean) / std
                return np.clip(normed, -3.0, 3.0)
        return volume
    return volume


def load_mri_file(file_path):
    """
    Loads brain MRI scan from various formats:
    - NIfTI (.nii, .nii.gz)
    - DICOM (.dcm)
    - Image slices (.png, .jpg, .jpeg)
    - NumPy arrays (.npy)
    Returns: 2D or 3D numpy array normalized to [0, 1].
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in [".nii", ".gz"]:
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required to load NIfTI (.nii/.nii.gz) files. Install via pip install nibabel.")
        img = nib.load(str(path))
        data = img.get_fdata()
        return normalize_intensity(data)

    elif suffix == ".dcm":
        if not HAS_PYDICOM:
            raise ImportError("pydicom is required to load DICOM (.dcm) files. Install via pip install pydicom.")
        dcm = pydicom.dcmread(str(path))
        data = dcm.pixel_array.astype(np.float32)
        return normalize_intensity(data)

    elif suffix == ".npy":
        data = np.load(str(path))
        return normalize_intensity(data)

    elif suffix in [".png", ".jpg", ".jpeg"]:
        img = Image.open(path).convert("L")
        data = np.array(img, dtype=np.float32)
        return normalize_intensity(data)

    else:
        raise ValueError(f"Unsupported MRI file format: {suffix}")


def extract_brain_slices(volume_3d, plane="axial", num_slices=3):
    """
    Extracts representative central slices from a 3D MRI volume.
    plane: 'axial', 'coronal', or 'sagittal'
    """
    if volume_3d.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume_3d.shape}")

    if plane == "axial":
        axis_len = volume_3d.shape[2]
        center = axis_len // 2
        indices = np.linspace(center - 10, center + 10, num_slices, dtype=int)
        return [volume_3d[:, :, idx] for idx in indices]

    elif plane == "coronal":
        axis_len = volume_3d.shape[1]
        center = axis_len // 2
        indices = np.linspace(center - 10, center + 10, num_slices, dtype=int)
        return [volume_3d[:, idx, :] for idx in indices]

    elif plane == "sagittal":
        axis_len = volume_3d.shape[0]
        center = axis_len // 2
        indices = np.linspace(center - 10, center + 10, num_slices, dtype=int)
        return [volume_3d[idx, :, :] for idx in indices]


def get_mri_transforms(image_size=(224, 224), is_train=True):
    """
    Data transformation and augmentation pipeline for MRI brain slices.
    Applies affine transformations, small rotations, and scaling.
    """
    if is_train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(image_size),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
