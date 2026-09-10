"""
EfficientNet-B3 transfer learning model for Pneumonia detection.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetDiagnosisModel(nn.Module):
    """
    EfficientNet-B3 based binary classifier for pneumonia detection.

    Uses timm's pretrained EfficientNet backbone with a custom
    classification head and optional dropout for regularization.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super(EfficientNetDiagnosisModel, self).__init__()

        # Load pretrained backbone
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier head
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimension from backbone
        in_features = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        output = self.classifier(features)
        return output

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate feature maps for Grad-CAM."""
        return self.backbone.forward_features(x)


def build_model(config: dict) -> nn.Module:
    """Factory function to build model from config dict."""
    return EfficientNetDiagnosisModel(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
        freeze_backbone=config["model"]["freeze_backbone"],
    )
