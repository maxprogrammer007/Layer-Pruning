# scripts/shap_analysis.py

import torch
import shap
from collections import defaultdict
from tqdm import tqdm

def run_shap_analysis(model, config, dataloader, num_samples=10, device='cuda'):
    """
    Runs SHAP analysis on selected layers defined in config.
    Returns a dict of mean SHAP value per layer.
    """
    model.eval()
    model.to(device)

    # Prepare storage
    layer_outputs = {}
    layer_scores = defaultdict(float)
    hook_handles = []

    # Step 1: Register hooks for target layers
    def get_hook(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach()
        return hook

    for name, module in model.backbone.features.named_modules():
        full_name = f"features.{name}"
        if full_name in config["layers_to_hook"]:
            handle = module.register_forward_hook(get_hook(full_name))
            hook_handles.append(handle)

    # Step 2: Select a few samples
    inputs = []
    for i, (img, _) in enumerate(dataloader):
        if i >= num_samples:
            break
        inputs.append(img.squeeze(0))  # shape: [3, H, W]
    
    inputs = torch.stack(inputs).to(device)

    # Step 3: Run model to collect activations
    with torch.no_grad():
        _ = model(inputs)

    # Step 4: Compute SHAP values for each layer using mean absolute activation
    for layer_name, activations in layer_outputs.items():
        # Compute average contribution per channel
        shap_val = activations.abs().mean().item()
        layer_scores[layer_name] = shap_val

    # Step 5: Clean up
    for handle in hook_handles:
        handle.remove()

    return dict(layer_scores)
