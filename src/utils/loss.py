
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss for ordered classification tasks like DR grading.
    Penalizes predictions that are further away from the true class more heavily.
    
    Implementation typically uses multiple binary classifiers:
    Rank 0: > Grade 0?
    Rank 1: > Grade 1?
    ...
    Rank K-1: > Grade K-1?
    
    However, adapting a standard 5-class output head for ordinal loss
    can be done using:
    Sum of binary cross entropy losses for each rank threshold.
    
    Or simpler approach for adaptation:
    Weighted Cross Entropy where weights correspond to distance matrix? No.
    
    Common approach for existing standard head (C classes):
    Convert label y to cumulative probability vector.
    e.g. y=2, NumClasses=5
    Target vector: [1, 1, 1, 0, 0] (Is >0? Yes. >1? Yes. >2? No.)
    
    But our model outputs 5 logits.
    We can treat this as K-1 binary tasks if we change the head, 
    OR we can use Earth Mover's Distance (EMD) / Squared Error on Softmax.
    
    Simpler "Soft" Ordinal Loss for standard CE Head:
    MSE between softmax prediction and one-hot label? No.
    
    Let's use the technique where we transform the single label into 
    K binary labels (like Coral Loss) but that requires model architectural change usually.
    
    Alternative: "Kappa Loss" or simpler "Distance Weighted Cross Entropy".
    
    Let's go with a robust Weighted MSE on standard logits (treating class as float)
    PLUS standard Cross Entropy.
    
    OR:
    Modified Cross Entropy: EMD-based.
    L_ordinal = Sum_k (p_k * |k - y|^2)
    
    Let's implement the EMD-based ordinal loss which works with standard softmax outputs.
    It penalizes probability mass assigned to distant classes.
    """
    def __init__(self, num_classes=5, alpha=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha # Weight for the ordinal term
        self.ce = nn.CrossEntropyLoss()
        
        # Default weights: 1:2:3:4:5 ratio to maximize sensitivity for higher grades
        if class_weights is None:
            self.class_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
        else:
            if len(class_weights) != num_classes:
                raise ValueError(f"class_weights must have {num_classes} elements.")
            self.class_weights = class_weights
        
    def forward(self, logits, targets):
        """
        logits: [B, NumClasses]
        targets: [B] (LongTensor)
        """
        device = logits.device
        weights = torch.ones_like(targets, dtype=torch.float32, device=device)
        
        # Apply configured weights
        for i, w in enumerate(self.class_weights):
            weights[targets == i] = w
        
        # This creates an aggressive gradient signal that scales linearly with disease severity.
        
        # 1. Standard Cross Entropy (Weighted)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # Apply weights and mean
        ce_loss = (ce_loss * weights).mean()
        
        # 2. Ordinal Term (Earth Mover's Distance approximation)
        probs = F.softmax(logits, dim=1)
        
        # Class values: [0, 1, 2, 3, 4]
        idx = torch.arange(self.num_classes, dtype=torch.float32, device=device)
        
        # Expected prediction: Sum(p_i * i)
        expected_pred = torch.sum(probs * idx, dim=1)
        
        # MSE between Expected Prediction and True Label (Weighted)
        mse_term = F.mse_loss(expected_pred, targets.float(), reduction='none')
        mse_term = (mse_term * weights).mean()
        
        return ce_loss + self.alpha * mse_term
