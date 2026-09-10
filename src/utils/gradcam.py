"""
Grad-CAM implementation for visualizing model explanations on X-Ray images.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Visualizes which regions of the input image are most important
    for the model's prediction decision.

    Reference: https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: Trained PyTorch model.
            target_layer: The convolutional layer to hook (usually last conv layer).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, C, H, W).
            target_class: Class index to explain. If None, uses predicted class.

        Returns:
            cam: Normalized heatmap as numpy array of shape (H, W).
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Compute Grad-CAM
        gradients = self.gradients[0]           # (C, H, W)
        activations = self.activations[0]       # (C, H, W)

        weights = gradients.mean(dim=(1, 2))    # Global Average Pooling
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            original_image: Original image as numpy array (H, W, 3) in RGB.
            cam: Grad-CAM heatmap of shape (h, w).
            alpha: Transparency of overlay (0=original, 1=heatmap only).
            colormap: OpenCV colormap to use.

        Returns:
            Blended image as numpy array (H, W, 3) in RGB.
        """
        h, w = original_image.shape[:2]

        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (w, h))
        cam_uint8 = np.uint8(255 * cam_resized)

        # Apply colormap
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)

        return overlay
