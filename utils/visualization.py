"""
Visualization utilities for DenWide-Net.
Includes:
- Grad-CAM (Gradient-weighted Class Activation Mapping) for MRI biomarker explainability.
- Training Loss / Accuracy curves plotting.
- Confusion Matrix and ROC Curve visualization.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for DenWide-Net.
    Generates class-activation heatmaps identifying brain anatomical regions
    driving ASD diagnosis predictions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Computes 2D CAM heatmap for input_tensor (1, C, H, W).
        Returns: 2D numpy array normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pool the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input image size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max - c_min > 1e-6:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap(original_img, heatmap, alpha=0.45, colormap="jet"):
    """
    Overlays a Grad-CAM heatmap onto a 2D brain MRI scan.
    original_img: PIL Image or (H, W) or (H, W, 3) numpy array
    heatmap: (H, W) numpy array in range [0, 1]
    Returns: PIL Image of blended overlay
    """
    if isinstance(original_img, Image.Image):
        base_img = np.array(original_img.convert("RGB"))
    else:
        base_img = np.array(original_img)
        if base_img.ndim == 2:
            base_img = np.stack([base_img] * 3, axis=-1)
        if base_img.max() <= 1.0:
            base_img = (base_img * 255.0).astype(np.uint8)

    # Apply colormap using matplotlib
    cm = plt.get_cmap(colormap)
    colored_heatmap = cm(heatmap)[:, :, :3]  # RGB, float in [0, 1]
    colored_heatmap = (colored_heatmap * 255.0).astype(np.uint8)

    # Blend original MRI with heatmap
    blended = (1 - alpha) * base_img.astype(np.float32) + alpha * colored_heatmap.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


def plot_training_curves(history, save_path):
    """Plots and saves loss and accuracy curves across training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("DenWide-Net Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", color="#2ca02c", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("DenWide-Net Classification Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path):
    """Plots and saves confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Confusion Matrix - DenWide-Net", fontsize=12, fontweight="bold")
    plt.ylabel("True Diagnosis")
    plt.xlabel("Predicted Diagnosis")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """Plots and saves Receiver Operating Characteristic (ROC) curve."""
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#d62728", lw=2, label=f"DenWide-Net ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
