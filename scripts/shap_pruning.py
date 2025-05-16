# scripts/shap_pruning.py

import torch
import torch.nn as nn
from collections import defaultdict


def get_layer_hooks(model, layers_to_hook):
    """
    Register forward hooks to capture output activations from specific layers.
    """
    layer_outputs = {}
    hook_handles = []

    def make_hook(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach()
        return hook

    for name, module in model.backbone.features.named_children():
        full_name = f"features.{name}"
        if full_name in layers_to_hook:
            handle = module.register_forward_hook(make_hook(full_name))
            hook_handles.append(handle)

    return layer_outputs, hook_handles


def compute_shap_scores(model, dataloader, config, num_samples=10, device="cuda"):
    """
    Run SHAP-like analysis using mean absolute activations for each layer.
    """
    model.eval()
    model.to(device)

    layer_outputs, hooks = get_layer_hooks(model, config["layers_to_hook"])

    # Run few images through model
    with torch.no_grad():
        for i, (img, _) in enumerate(dataloader):
            if i >= num_samples:
                break
            img = img.to(device)
            _ = model([img[0]])

    # Compute SHAP-style scores
    shap_scores = {}
    for name, tensor in layer_outputs.items():
        score = tensor.abs().mean().item()
        shap_scores[name] = round(score, 6)

    # Cleanup
    for handle in hooks:
        handle.remove()

    return shap_scores


def prune_model_by_shap(model, shap_scores, threshold=0.2):
    """
    Prune layers with SHAP score below threshold by replacing with nn.Identity().
    """
    removed_layers = []

    for name, module in model.backbone.features.named_children():
        full_name = f"features.{name}"
        if full_name in shap_scores and shap_scores[full_name] < threshold:
            setattr(model.backbone.features, name, nn.Identity())
            removed_layers.append(full_name)

    return model, removed_layers
