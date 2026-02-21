"""
Explainability Module using Grad-CAM.
Generates heatmaps for Swin-Transformer backbone.
"""

import torch
import numpy as np
import cv2
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
from typing import Tuple

class ExplainabilityEngine:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Default target layer for Swin-Tiny (last stage block)
        # For MONAI 1.3.0 compatibility, use the global pooling layer
        if target_layer is None:
            # Use the global pooling layer which is simpler and guaranteed to exist
            print("Targeting Global Average Pooling layer...")
            self.target_layer = self.model.global_pool
        else:
            self.target_layer = target_layer
            
        self.cam = LayerGradCam(self.model, self.target_layer)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: (1, 3, H, W) normalized tensor
            target_class: Target class index (0-4). If None, uses predicted class.
            
        Returns:
            heatmap: (H, W) numpy array value 0-1
        """
        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                target_class = torch.argmax(logits, dim=1).item()
        
        # Generate CAM
        # Attribute takes the input tensor and the target class index
        # For Swin, the output of the target layer (B, H/32*W/32, C) or (B, C, H/32, W/32) depends on reshape
        # MONAI Swin outputs are reshaped in forward but the blocks might be different.
        # Captum expects standard forward.
        
        # Note: Swin Transformer usually handles 2D spatial as flattened tokens in intermediate layers.
        # LayerGradCam might fail if the layer output is 3D (B, Tokens, C).
        # We need to reshape the attribution result inside the hook or post-process.
        # But LayerGradCam computes gradients w.r.t layer output.
        
        try:
            attributions = self.cam.attribute(input_tensor, target=target_class, relu_attributions=True)
            
            # Swin Output often (B, HxW, C) or (B, C, H, W)
            # Check shape
            # If (B, Tokens, C), we need to reshape to (B, C, Ht, Wt) then avg over C
            
            attr = attributions.detach().cpu()
            
            if attr.dim() == 3: # (B, Tokens, C)
                # Reshape tokens to 2D
                # Assuming square input 224x224 -> 7x7 windows * ?
                # Swin T: 4 stages. Stage 4 resolution is H/32 = 7.
                # So 49 tokens.
                b, n_tokens, c = attr.shape
                h = w = int(np.sqrt(n_tokens))
                attr = attr.permute(0, 2, 1).view(b, c, h, w)
            
            # Upsample to input size
            attr = LayerGradCam.interpolate(attr, input_tensor.shape[2:])
            
            # Normalize to 0-1
            heatmap = attr.squeeze().cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            return heatmap
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            # Return empty heatmap on failure
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

    @staticmethod
    def overlay_heatmap(original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: (H, W, 3) RGB numpy array [0-255]
            heatmap: (H, W) numpy array [0-1]
            alpha: Opacity of heatmap
            
        Returns:
            overlay: (H, W, 3) RGB numpy array
        """
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Apply colormap (JET)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
        return overlay
