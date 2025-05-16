# scripts/model_loader.py

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn

def load_model(model_name: str):
    """
    Load a torchvision detection model by name.
    Extend this with more models as needed.
    """
    if model_name == "mobilenet_ssd":
        model = ssdlite320_mobilenet_v3_large(pretrained=True)
    elif model_name == "resnet50_frcnn":
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device
