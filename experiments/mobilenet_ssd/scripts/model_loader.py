# experiments/mobilenet_ssd/scripts/model_loader.py
import torch

def load_model(model_cfg, device):
    """
    model_cfg: dict with keys 'type' and 'checkpoint'
    device: torch device string e.g. 'cuda'
    """
    if model_cfg['type'] == 'mobilenet_ssd':
        # Replace this with your actual import/path
        from detectors.mobilenet_ssd import MobileNetSSD  
        model = MobileNetSSD()
    else:
        raise ValueError(f"Unsupported model type {model_cfg['type']}")
    state = torch.load(model_cfg['checkpoint'], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
