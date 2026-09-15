"""
DenWide-Net (Dense WideResNet) Architecture for Autism Spectrum Disorder (ASD) Detection.

Reference:
A. Arunkumar & S. Sudhakar Ilango, "DenWide-Net: Dense WideresNet for autism
spectrum disorder detection using MRI image", Biomedical Signal Processing and Control.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import (
    DenseWideBlock,
    TransitionLayer,
    SpatialAttentionGate,
    DenseWideBlock3D,
    TransitionLayer3D
)


class DenWideNet(nn.Module):
    """
    DenWide-Net: Dense WideResNet for 2D Brain MRI Slice Classification.
    Combines dense feature reuse across layers with wide residual representations
    and spatial attention mechanism to detect ASD neuroanatomical biomarkers.
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        growth_rate=24,
        widen_factor=2,
        compression_factor=0.5,
        block_config=(4, 6, 8, 4),
        dropout_rate=0.2,
        use_attention=True
    ):
        super().__init__()
        self.use_attention = use_attention
        init_features = growth_rate * 2

        # 1. Stem / Initial Feature Extractor
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Dense-Wide Stages
        num_features = init_features
        self.stage1 = DenseWideBlock(
            num_layers=block_config[0],
            in_channels=num_features,
            growth_rate=growth_rate,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate
        )
        num_features = self.stage1.out_channels
        self.trans1 = TransitionLayer(num_features, compression_factor=compression_factor)
        num_features = self.trans1.out_channels

        self.stage2 = DenseWideBlock(
            num_layers=block_config[1],
            in_channels=num_features,
            growth_rate=growth_rate,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate
        )
        num_features = self.stage2.out_channels
        self.trans2 = TransitionLayer(num_features, compression_factor=compression_factor)
        num_features = self.trans2.out_channels

        self.stage3 = DenseWideBlock(
            num_layers=block_config[2],
            in_channels=num_features,
            growth_rate=growth_rate,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate
        )
        num_features = self.stage3.out_channels
        self.trans3 = TransitionLayer(num_features, compression_factor=compression_factor)
        num_features = self.trans3.out_channels

        self.stage4 = DenseWideBlock(
            num_layers=block_config[3],
            in_channels=num_features,
            growth_rate=growth_rate,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate
        )
        num_features = self.stage4.out_channels

        # 3. Post-Stage Batch Normalization & Attention
        self.post_bn = nn.BatchNorm2d(num_features)
        self.post_relu = nn.ReLU(inplace=True)

        if use_attention:
            self.attention = SpatialAttentionGate(num_features)
        else:
            self.attention = nn.Identity()

        # 4. Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Initialize Weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        """Extract multi-scale deep features prior to pooling."""
        x = self.stem(x)
        x = self.trans1(self.stage1(x))
        x = self.trans2(self.stage2(x))
        x = self.trans3(self.stage3(x))
        x = self.stage4(x)
        x = self.post_relu(self.post_bn(x))
        x = self.attention(x)
        return x

    def forward(self, x):
        features = self.extract_features(x)
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, 1)
        logits = self.classifier(flattened)
        return logits

    def predict_proba(self, x):
        """Returns softmax probabilities for ASD and Control classes."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def get_cam_target_layer(self):
        """Target layer for Grad-CAM saliency map visualization."""
        return self.stage4


class DenWideNet3D(nn.Module):
    """
    3D DenWide-Net for Volumetric Neuroimaging (3D Structural MRI / fMRI).
    Processes complete 3D brain volumes (e.g. 1 x 64 x 64 x 64 or 1 x 128 x 128 x 128).
    """
    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        growth_rate=16,
        widen_factor=2,
        compression_factor=0.5,
        block_config=(3, 4, 6, 3),
        dropout_rate=0.2
    ):
        super().__init__()
        init_features = growth_rate * 2

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        num_features = init_features
        self.stage1 = DenseWideBlock3D(block_config[0], num_features, growth_rate, widen_factor, dropout_rate)
        num_features = self.stage1.out_channels
        self.trans1 = TransitionLayer3D(num_features, compression_factor)
        num_features = self.trans1.out_channels

        self.stage2 = DenseWideBlock3D(block_config[1], num_features, growth_rate, widen_factor, dropout_rate)
        num_features = self.stage2.out_channels
        self.trans2 = TransitionLayer3D(num_features, compression_factor)
        num_features = self.trans2.out_channels

        self.stage3 = DenseWideBlock3D(block_config[2], num_features, growth_rate, widen_factor, dropout_rate)
        num_features = self.stage3.out_channels
        self.trans3 = TransitionLayer3D(num_features, compression_factor)
        num_features = self.trans3.out_channels

        self.stage4 = DenseWideBlock3D(block_config[3], num_features, growth_rate, widen_factor, dropout_rate)
        num_features = self.stage4.out_channels

        self.post_bn = nn.BatchNorm3d(num_features)
        self.post_relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.trans1(self.stage1(x))
        x = self.trans2(self.stage2(x))
        x = self.trans3(self.stage3(x))
        x = self.stage4(x)
        x = self.post_relu(self.post_bn(x))
        pooled = self.global_pool(x)
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)


def build_denwidenet(
    in_channels=3,
    num_classes=2,
    growth_rate=24,
    widen_factor=2,
    dropout_rate=0.2,
    use_attention=True,
    is_3d=False
):
    """Factory function to build a DenWide-Net instance."""
    if is_3d:
        return DenWideNet3D(
            in_channels=1,
            num_classes=num_classes,
            growth_rate=growth_rate,
            widen_factor=widen_factor,
            dropout_rate=dropout_rate
        )
    return DenWideNet(
        in_channels=in_channels,
        num_classes=num_classes,
        growth_rate=growth_rate,
        widen_factor=widen_factor,
        dropout_rate=dropout_rate,
        use_attention=use_attention
    )
