"""
Architectural components for DenWide-Net (Dense WideResNet).
Combines dense feature reuse (DenseNet) with expanded channel capacity
and residual shortcuts (WideResNet) alongside spatial attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Basic Convolution + Batch Normalization + ReLU / LeakyReLU block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class WideResidualUnit(nn.Module):
    """
    Wide Residual Unit (WideResNet concept) with widening factor k.
    Features expanded channels, pre-activation or post-activation, and residual shortcut.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + residual


class DenseWideLayer(nn.Module):
    """
    Dense-Wide Layer:
    1. Bottleneck 1x1 conv to compress aggregated dense inputs.
    2. Wide Residual Branch (widened features with residual connections).
    3. Output channel dimension matches growth_rate * widen_factor.
    """
    def __init__(self, in_channels, growth_rate, widen_factor=2, dropout_rate=0.2):
        super().__init__()
        inter_channels = growth_rate * 4
        wide_channels = growth_rate * widen_factor

        # Bottleneck 1x1 Conv
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        # Wide Residual Unit
        self.wide_unit = WideResidualUnit(inter_channels, wide_channels, dropout_rate=dropout_rate)

        # Final projection to match growth_rate for dense concatenation
        self.out_conv = nn.Conv2d(wide_channels, growth_rate, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.wide_unit(out)
        out = self.out_conv(out)
        return out


class DenseWideBlock(nn.Module):
    """
    Dense-Wide Block (DWB):
    Contains multiple DenseWideLayers where each layer's output is concatenated
    with all previous feature maps.
    """
    def __init__(self, num_layers, in_channels, growth_rate, widen_factor=2, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            layer = DenseWideLayer(
                in_channels=current_channels,
                growth_rate=growth_rate,
                widen_factor=widen_factor,
                dropout_rate=dropout_rate
            )
            self.layers.append(layer)
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition Layer between Dense-Wide Blocks.
    Reduces feature map spatial dimension (2x2 AvgPool) and compresses channels (1x1 Conv).
    """
    def __init__(self, in_channels, compression_factor=0.5):
        super().__init__()
        out_channels = int(math.floor(in_channels * compression_factor))
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.block(x)


class SpatialAttentionGate(nn.Module):
    """
    Spatial Attention Gate to focus on discriminative neuroanatomical regions
    in brain MRI (e.g. corpus callosum, amygdala, cerebellar vermis).
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.conv(x)
        return x * mask


# =====================================================================
# 3D Components for Volumetric MRI (NIfTI / DICOM full volumes)
# =====================================================================

class WideResidualUnit3D(nn.Module):
    """3D Wide Residual Unit for volumetric brain scans."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        return out + residual


class DenseWideLayer3D(nn.Module):
    """3D Dense-Wide Layer for volumetric brain features."""
    def __init__(self, in_channels, growth_rate, widen_factor=2, dropout_rate=0.2):
        super().__init__()
        inter_channels = growth_rate * 4
        wide_channels = growth_rate * widen_factor

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.wide_unit = WideResidualUnit3D(inter_channels, wide_channels, dropout_rate=dropout_rate)
        self.out_conv = nn.Conv3d(wide_channels, growth_rate, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.wide_unit(out)
        out = self.out_conv(out)
        return out


class DenseWideBlock3D(nn.Module):
    """3D Dense-Wide Block."""
    def __init__(self, num_layers, in_channels, growth_rate, widen_factor=2, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            layer = DenseWideLayer3D(
                in_channels=current_channels,
                growth_rate=growth_rate,
                widen_factor=widen_factor,
                dropout_rate=dropout_rate
            )
            self.layers.append(layer)
            current_channels += growth_rate
        self.out_channels = current_channels

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)


class TransitionLayer3D(nn.Module):
    """3D Transition Layer with channel compression and 3D average pooling."""
    def __init__(self, in_channels, compression_factor=0.5):
        super().__init__()
        out_channels = int(math.floor(in_channels * compression_factor))
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.block(x)
