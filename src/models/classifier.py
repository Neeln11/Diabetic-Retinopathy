"""
Swin Transformer-based Diabetic Retinopathy Classifier
Optimized for 4GB VRAM with gradient checkpointing and mixed precision support.

Medical-grade classifier for 5-class DR grading (0-4):
- 0: No DR
- 1: Mild NPDR
- 2: Moderate NPDR
- 3: Severe NPDR
- 4: Proliferative DR
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from typing import Optional, Tuple


class SwinDRClassifier(nn.Module):
    """
    Swin Transformer-based DR Classifier optimized for 4GB VRAM.
    
    Architecture:
        - Swin-Tiny backbone (28M parameters)
        - Global Average Pooling
        - Dropout (0.3)
        - Linear classifier (5 classes)
    
    VRAM Optimizations:
        - Gradient checkpointing enabled
        - Mixed precision (FP16) support
        - Efficient forward pass
        - No metadata tracking
    
    Args:
        img_size: Input image size (default: 224)
        num_classes: Number of DR grades (default: 5 for grades 0-4)
        feature_size: Base feature dimension (default: 48 for VRAM efficiency)
        use_checkpoint: Enable gradient checkpointing (default: True)
        dropout_rate: Dropout rate before classifier (default: 0.3)
        pretrained: Load pretrained weights if available (default: False)
    
    Input:
        - x: (B, 3, 224, 224) RGB fundus images after BenGrahamPreprocessingD
    
    Output:
        - logits: (B, 5) class logits for DR grades 0-4
    
    Example:
        >>> model = SwinDRClassifier(use_checkpoint=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([1, 5])
    """
    
    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 5,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        dropout_rate: float = 0.3,
        pretrained: bool = False,
        ordinal_head: bool = False,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.use_checkpoint = use_checkpoint
        self.ordinal_head = ordinal_head
        
        # Swin-Tiny configuration for 4GB VRAM
        # Standard Swin-T: depths=[2,2,6,2], num_heads=[3,6,12,24]
        # We use feature_size=48 (vs 96 in standard) to reduce VRAM
        # Note: MONAI 1.3 uses img_size parameter
        self.backbone = SwinUNETR(
            img_size=(img_size, img_size),
            in_channels=3,
            out_channels=num_classes,  # Not used, we'll use custom head
            feature_size=feature_size,
            depths=[2, 2, 6, 2],  # Swin-Tiny depths
            num_heads=[3, 6, 12, 24],  # Swin-Tiny attention heads
            use_checkpoint=use_checkpoint,  # Critical for VRAM
            spatial_dims=2,  # 2D images
        )
        
        # Extract only the Swin encoder (not the full UNETR decoder)
        self.encoder = self.backbone.swinViT
        
        # Calculate output feature dimension
        # Swin-T final stage: feature_size * 16 = 48 * 16 = 768
        # (This is because of the hierarchical downsampling in Swin)
        self.feature_dim = feature_size * 16
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        out_features = num_classes - 1 if ordinal_head else num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, out_features)
        )
        
        # Initialize weights
        self._init_weights()
        
        if pretrained:
            self._load_pretrained()
    
    def _init_weights(self):
        """Initialize classifier weights with Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _load_pretrained(self):
        """Load pretrained weights if available."""
        # TODO: Implement pretrained weight loading
        # This would load ImageNet or medical imaging pretrained weights
        print("Warning: Pretrained weights not implemented yet")
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            logits: Class logits of shape (B, num_classes)
        """
        # Input validation
        assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D"
        assert x.size(1) == 3, f"Expected 3 channels, got {x.size(1)}"
        
        # Extract features using Swin encoder
        # Returns hidden states from all stages
        hidden_states = self.encoder(x)
        
        # Use features from the final stage (highest level)
        # hidden_states is a list: [stage1, stage2, stage3, stage4]
        features = hidden_states[-1]  # (B, feature_dim, H/32, W/32)
        
        # Global average pooling
        pooled = self.global_pool(features)  # (B, feature_dim, 1, 1)
        
        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Extract attention maps for explainability (Grad-CAM, attention visualization).
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            logits: Class predictions
            attention_maps: List of attention maps from each Swin stage
        """
        # This is a placeholder for future explainability features
        # Will be used for generating attention visualizations
        hidden_states = self.encoder(x)
        features = hidden_states[-1]
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        
        return logits, hidden_states
    
    def count_parameters(self) -> dict:
        """
        Count model parameters for VRAM estimation.
        
        Returns:
            dict with total, trainable, and non-trainable parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params,
            'total_millions': total_params / 1e6,
        }
    
    def estimate_vram(self, batch_size: int = 1, precision: str = 'fp32') -> dict:
        """
        Estimate VRAM usage for training.
        
        Args:
            batch_size: Training batch size
            precision: 'fp32' or 'fp16'
        
        Returns:
            dict with VRAM estimates in GB
        """
        params = self.count_parameters()['total']
        
        # Bytes per parameter
        bytes_per_param = 4 if precision == 'fp32' else 2
        
        # Model weights
        model_size = params * bytes_per_param
        
        # Gradients (same size as weights)
        gradient_size = model_size
        
        # Optimizer state (Adam: 2x weights for momentum and variance)
        optimizer_size = 2 * model_size
        
        # Activations (rough estimate, depends on checkpoint)
        if self.use_checkpoint:
            # With checkpointing, only store activations for current layer
            activation_size = batch_size * self.feature_dim * (self.img_size // 32) ** 2 * bytes_per_param * 4
        else:
            # Without checkpointing, store all intermediate activations
            activation_size = batch_size * self.feature_dim * (self.img_size // 32) ** 2 * bytes_per_param * 20
        
        # Input batch
        input_size = batch_size * 3 * self.img_size * self.img_size * bytes_per_param
        
        total_bytes = model_size + gradient_size + optimizer_size + activation_size + input_size
        total_gb = total_bytes / (1024 ** 3)
        
        return {
            'model_gb': model_size / (1024 ** 3),
            'gradients_gb': gradient_size / (1024 ** 3),
            'optimizer_gb': optimizer_size / (1024 ** 3),
            'activations_gb': activation_size / (1024 ** 3),
            'input_gb': input_size / (1024 ** 3),
            'total_gb': total_gb,
            'precision': precision,
            'batch_size': batch_size,
            'checkpoint_enabled': self.use_checkpoint,
        }


def create_swin_dr_classifier(
    vram_optimized: bool = True,
    **kwargs
) -> SwinDRClassifier:
    """
    Factory function to create SwinDRClassifier with recommended settings.
    
    Args:
        vram_optimized: If True, use 4GB VRAM optimized settings
        **kwargs: Additional arguments passed to SwinDRClassifier
    
    Returns:
        SwinDRClassifier instance
    
    Example:
        >>> # For 4GB VRAM
        >>> model = create_swin_dr_classifier(vram_optimized=True)
        >>> 
        >>> # For larger VRAM (8GB+)
        >>> model = create_swin_dr_classifier(
        ...     vram_optimized=False,
        ...     feature_size=96,
        ...     use_checkpoint=False
        ... )
    """
    if vram_optimized:
        # 4GB VRAM optimized settings
        default_kwargs = {
            'img_size': 224,
            'feature_size': 48,
            'use_checkpoint': True,
            'dropout_rate': 0.3,
        }
    else:
        # Standard settings for larger VRAM
        default_kwargs = {
            'img_size': 384,
            'feature_size': 96,
            'use_checkpoint': False,
            'dropout_rate': 0.3,
        }
    
    # Override defaults with user-provided kwargs
    default_kwargs.update(kwargs)
    
    return SwinDRClassifier(**default_kwargs)


if __name__ == "__main__":
    # Quick test
    print("Creating Swin-DR Classifier...")
    model = create_swin_dr_classifier(vram_optimized=True)
    
    print("\nModel Statistics:")
    params = model.count_parameters()
    print(f"Total parameters: {params['total_millions']:.2f}M")
    print(f"Trainable parameters: {params['trainable'] / 1e6:.2f}M")
    
    print("\nVRAM Estimates:")
    vram_fp32 = model.estimate_vram(batch_size=1, precision='fp32')
    vram_fp16 = model.estimate_vram(batch_size=1, precision='fp16')
    
    print(f"\nFP32 (batch_size=1):")
    print(f"  Total VRAM: {vram_fp32['total_gb']:.2f} GB")
    print(f"  Model: {vram_fp32['model_gb']:.3f} GB")
    print(f"  Activations: {vram_fp32['activations_gb']:.3f} GB")
    
    print(f"\nFP16 (batch_size=1):")
    print(f"  Total VRAM: {vram_fp16['total_gb']:.2f} GB")
    print(f"  Model: {vram_fp16['model_gb']:.3f} GB")
    print(f"  Activations: {vram_fp16['activations_gb']:.3f} GB")
    
    print("\nTesting forward pass...")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output logits: {logits[0].tolist()}")
    
    print("\n[SUCCESS] Model created successfully!")
