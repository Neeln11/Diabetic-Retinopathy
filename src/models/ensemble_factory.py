
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, densenet121, EfficientNet_B0_Weights, DenseNet121_Weights
from src.models.classifier import create_swin_dr_classifier
import gc
import os

class MetaPredictor:
    """
    3-Model Ensemble (Swin + EfficientNet + DenseNet) with VRAM-safe Sequential Inference.
    
    Strategy:
        1. Load Model -> Predict -> Unload -> Clear Cache
        2. Repeat for all 3 models
        3. Simple Weighted Average of Probabilities
        
    Weights:
        - Swin-Tiny (Adapted): 0.4
        - EfficientNet-B0 (ImageNet): 0.3
        - DenseNet-121 (ImageNet): 0.3
    """
    
    def __init__(self, swin_path=None, effnet_path=None, densenet_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.swin_path = swin_path
        self.effnet_path = effnet_path
        self.densenet_path = densenet_path
        self.device = device
        self.weights = {'swin': 0.4, 'effnet': 0.3, 'densenet': 0.3}
        
    def _load_swin(self):
        """Load the domain-adapted Swin Transformer"""
        print(f">>> Loading Swin-Tiny from {self.swin_path}...")
        model = create_swin_dr_classifier(vram_optimized=True, num_classes=5)
        if self.swin_path and os.path.exists(self.swin_path):
            state_dict = torch.load(self.swin_path, map_location='cpu')
            # Handle potential DataParallel wrapper
            if 'module.' in list(state_dict.keys())[0]:
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            print("!!! Warning: Swin weights not found! Using random init.")
            
        model.to(self.device).eval()
        return model

    def _load_effnet(self):
        """Load EfficientNet-B0 (Pretrained on ImageNet or Fine-tuned)"""
        print(">>> Loading EfficientNet-B0...")
        # Use simple 5-class head adapter
        model = efficientnet_b0(weights=None) # Initialize without weights first
        model.classifier[1] = nn.Linear(1280, 5) # Replace 1000-class head
        
        if self.effnet_path and os.path.exists(self.effnet_path):
            print(f"   -> Loading weights from {self.effnet_path}")
            model.load_state_dict(torch.load(self.effnet_path, map_location='cpu'))
        else:
            print("   -> Using ImageNet weights (Pretrained)")
            # Load ImageNet weights if no custom path (strict=False to ignore head mismatch if we loaded full model before? No, standard torchvision generic load)
            # Actually, cleanest is:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(1280, 5)
            
        model.to(self.device).eval()
        return model
        model.to(self.device).eval()
        return model

    def _load_densenet(self):
        """Load DenseNet-121 (Pretrained on ImageNet or Fine-tuned)"""
        print(">>> Loading DenseNet-121...")
        model = densenet121(weights=None)
        model.classifier = nn.Linear(1024, 5)
        
        if self.densenet_path and os.path.exists(self.densenet_path):
            print(f"   -> Loading weights from {self.densenet_path}")
            model.load_state_dict(torch.load(self.densenet_path, map_location='cpu'))
        else:
            print("   -> Using ImageNet weights (Pretrained)")
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            model.classifier = nn.Linear(1024, 5)
            
        model.to(self.device).eval()
        return model

    def predict_sequential(self, image_tensor):
        """
        Run inference sequentially to save VRAM.
        image_tensor: (B, 3, 224, 224)
        """
        predictions = {}
        T = 1.764 # Temperature Scaling for Calibration
        
        # 1. Swin Transformer
        try:
            model = self._load_swin()
            with torch.no_grad():
                logits = model(image_tensor.to(self.device))
                probs_swin = torch.softmax(logits / T, dim=1).cpu()
            predictions['swin'] = probs_swin
        except Exception as e:
            print(f"!!! Swin Failed: {e}")
            predictions['swin'] = torch.zeros(1, 5) # Fail safe
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # 2. EfficientNet
        try:
            model = self._load_effnet()
            with torch.no_grad():
                logits = model(image_tensor.to(self.device))
                probs_eff = torch.softmax(logits / T, dim=1).cpu()
            predictions['effnet'] = probs_eff
        except Exception as e:
            print(f"!!! EffNet Failed: {e}")
            predictions['effnet'] = torch.zeros(1, 5)
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # 3. DenseNet
        try:
            model = self._load_densenet()
            with torch.no_grad():
                logits = model(image_tensor.to(self.device))
                probs_dense = torch.softmax(logits / T, dim=1).cpu()
            predictions['densenet'] = probs_dense
        except Exception as e:
            print(f"!!! DenseNet Failed: {e}")
            predictions['densenet'] = torch.zeros(1, 5)
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        # Weighted Ensemble
        # Swin (0.4) + EffNet (0.3) + DenseNet (0.3)
        final_probs = (
            predictions['swin'] * self.weights['swin'] + 
            predictions['effnet'] * self.weights['effnet'] + 
            predictions['densenet'] * self.weights['densenet']
        )
        
        # Grey-Zone Logic
        final_class = torch.argmax(final_probs, dim=1).item()
        confidence = final_probs[0, final_class].item()
        is_ambiguous = 0.40 <= confidence <= 0.60
        
        # Get individual predictions for UI display
        breakdown = {
            'Swin': torch.argmax(predictions['swin'], dim=1).item(),
            'Swin_Probs': predictions['swin'],
            'EfficientNet': torch.argmax(predictions['effnet'], dim=1).item(),
            'DenseNet': torch.argmax(predictions['densenet'], dim=1).item(),
            'Ensemble': final_class,
            'Ensemble_Probs': final_probs,
            'Confidence': confidence,
            'Ambiguous_Flag': is_ambiguous,
            'Message': 'Ambiguous - Requires Doctor Review' if is_ambiguous else 'Confident'
        }
        
        return final_probs, breakdown

    def predict(self, image_path_or_tensor, return_breakdown=False):
        """
        Convenience method to predict from usage path or tensor.
        Returns the integer class label (0-4).
        """
        if isinstance(image_path_or_tensor, str) or isinstance(image_path_or_tensor, Path):
            # Load and preprocess
            from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ToTensord
            from src.preprocessing.transforms import BenGrahamPreprocessingD
            
            transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Resized(keys=["image"], spatial_size=(224, 224)), # Swin size
                BenGrahamPreprocessingD(keys=["image"], sigma=10),
                ScaleIntensityd(keys=["image"]),
                ToTensord(keys=["image"])
            ])
            
            data = {"image": str(image_path_or_tensor)}
            data = transforms(data)
            img_tensor = data["image"].unsqueeze(0)
        else:
            img_tensor = image_path_or_tensor
            
        final_probs, breakdown = self.predict_sequential(img_tensor)
        
        if return_breakdown:
            return breakdown
        return breakdown['Ensemble']
