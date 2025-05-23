# detectors/__init__.py
# This file makes detectors/ a Python package.

# detectors/mobilenet_ssd.py
import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

class MobileNetSSD(nn.Module):
    """
    Single Shot Detection model using MobileNet backbone.
    Uses torchvision's SSDLite with MobileNetV3.
    """
    def __init__(self, num_classes=21):
        super().__init__()
        # Instantiate SSDLite with MobileNetV3 backbone
        self.model = ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=num_classes)
        
        # Collect conv layer names for pruning
        self.layer_names = [
            name for name, module in self.model.named_modules()
            if isinstance(module, nn.Conv2d)
        ]

    def forward(self, images):
        # torchvision detection models expect a list of tensors
        if isinstance(images, torch.Tensor):
            images = list(images)
        return self.model(images)
