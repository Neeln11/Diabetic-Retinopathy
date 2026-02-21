import cv2
import numpy as np
import torch
from monai.transforms import MapTransform

class BenGrahamPreprocessingD(MapTransform):
    def __init__(self, keys, sigma=10):
        super().__init__(keys)
        self.sigma = sigma

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            
            # Convert generic tensor/metatensor to numpy
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            elif hasattr(img, "numpy"):
                img = img.numpy()
            
            # Ensure proper layout for OpenCV: (H, W, C)
            # MONAI usually gives (C, H, W)
            channel_first = False
            if img.ndim == 3 and img.shape[0] in [1, 3]:  # Likely (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)
                channel_first = True
            
            # Ensure uint8 [0, 255] for OpenCV operations
            is_float = False
            if img.dtype != np.uint8:
                is_float = True
                # Check if range is [0, 1]
                if img.max() <= 1.05:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Ben Graham's logic: img = 4 * original - 4 * blurred + 128
            # GaussianBlur expects (H, W, C)
            blurred = cv2.GaussianBlur(img, (0, 0), self.sigma)
            img = cv2.addWeighted(img, 4, blurred, -4, 128)
            
            # Convert back to float [0, 1] if input was float (or expected to be)
            # Actually, standard BenGraham output keeps the 128 offset which centers it
            # But typically we want standard scaling after.
            # Let's return typical BenGraham result (0-255 range usually)
            
            # Restore layout
            if channel_first:
                img = np.transpose(img, (2, 0, 1))  # -> (C, H, W)
            
            d[key] = img
        return d
