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


# scripts/shap_pruning.py

import torch.nn as nn

def prune_model_by_shap(model, shap_scores, threshold=0.2):
    """
    Replace low-contributing layers with Identity safely.
    Skips layers that would break the model shape-wise.
    """
    removed_layers = []
    skip_layers = set()

    # Step 1: Run one forward pass to record input/output shapes
    shape_cache = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            shape_cache[name] = {
                "in": inp[0].shape if isinstance(inp, tuple) else inp.shape,
                "out": out.shape
            }
        return hook

    for name, module in model.backbone.features.named_children():
        full_name = f"features.{name}"
        hooks.append(module.register_forward_hook(make_hook(full_name)))

    # Dummy pass
    import torch
    dummy = torch.randn(1, 3, 320, 320).to(next(model.parameters()).device)
    _ = model.backbone(dummy)

    for h in hooks:
        h.remove()

    # Step 2: Prune safely
    for name, module in model.backbone.features.named_children():
        full_name = f"features.{name}"
        if full_name in shap_scores and shap_scores[full_name] < threshold:
            if full_name not in shape_cache:
                continue  # no shape info
            in_c = shape_cache[full_name]["in"][1]
            out_c = shape_cache[full_name]["out"][1]
            if in_c != out_c:
                print(f"⚠️ Skipping {full_name}: shape mismatch {in_c} → {out_c}")
                continue  # critical layer
            setattr(model.backbone.features, name, nn.Identity())
            removed_layers.append(full_name)

    return model, removed_layers
