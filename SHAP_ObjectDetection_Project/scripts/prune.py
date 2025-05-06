# scripts/prune.py

import torch.nn as nn

def prune_layers(model, config, pruned_layer_names):
    """
    Replaces selected layers in the model with nn.Identity().
    Works only for layers inside `model.backbone.features` as per config.
    
    Returns:
        model: pruned model
        layers_removed: list of actually removed layer names
    """
    removed = []

    for name, module in model.backbone.features.named_children():
        full_name = f"features.{name}"
        if full_name in pruned_layer_names:
            setattr(model.backbone.features, name, nn.Identity())
            removed.append(full_name)
            print(f"✅ Pruned {full_name} → nn.Identity()")

    return model, removed
