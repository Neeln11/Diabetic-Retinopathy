
import torch
import torch.nn as nn
import torch.nn.functional as F

class CORALLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) for Ordinal Regression.
    Converts a single label 'y' containing K classes [0, 1, ..., K-1]
    into K-1 binary classification tasks.
    
    e.g., for 5 classes, label 2 maps to: [1, 1, 0, 0]
    This explicitly penalizes the distance between predicted rank and true grade.
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        logits: [B, num_classes - 1]
        targets: [B] (LongTensor) containing grades 0 to num_classes-1
        """
        device = logits.device
        batch_size = logits.size(0)
        
        # Convert targets to extended binary format (levels)
        # e.g., target 2 -> [1, 1, 0, 0]
        levels = torch.zeros(batch_size, self.num_classes - 1, dtype=torch.float32, device=device)
        for i in range(batch_size):
            if targets[i] > 0:
                levels[i, :targets[i]] = 1.0
                
        # Calculate Coral Loss (Sum of Binary Cross Entropy across ranks)
        # using log-sigmoid formulation for numerical stability
        val = (-torch.sum((F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (1 - levels)), dim=1))
        
        return torch.mean(val)

    @staticmethod
    def predict_labels(logits):
        """Convert ordinal logits to class predictions."""
        # Probabilities of exceeding each rank threshold
        probs = torch.sigmoid(logits)
        # Predict class by counting how many thresholds were exceeded (prob > 0.5)
        predicts = torch.sum(probs > 0.5, dim=1)
        return predicts

