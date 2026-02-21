import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

def get_segmentor(num_classes=2):
    """
    Returns a DeepLabv3+ model instance for lesion detection.
    """
    model = deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model
