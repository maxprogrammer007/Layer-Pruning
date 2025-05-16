# scripts/pruning_methods.py

import torch.nn as nn
import random

def l1_norm_prune(model, config, dataloader, threshold=0.1, device="cuda"):
    """
    Prune layers with lowest L1-norm based on output activation.
    """
    model.eval()
    model.to(device)
    l1_scores = {}

    def make_hook(name):
        def hook(module, input, output):
            l1_scores[name] = output.abs().sum().item()
        return hook

    hooks = []
    for name, module in model.backbone.features.named_children():
        full_name = f"features.{name}"
        if full_name in config["layers_to_hook"]:
            hooks.append(module.register_forward_hook(make_hook(full_name)))

    # Run one sample through
    for img, _ in dataloader:
        _ = model([img[0].to(device)])
        break

    for h in hooks:
        h.remove()

    # Normalize & prune
    max_score = max(l1_scores.values())
    to_prune = [k for k, v in l1_scores.items() if v / max_score < threshold]

    removed = []
    for name in to_prune:
        layer_id = name.split(".")[-1]
        setattr(model.backbone.features, layer_id, nn.Identity())
        removed.append(name)

    return model, removed


def random_prune(model, config, fraction=0.3):
    """
    Randomly prune a fraction of layers from the config list.
    """
    layer_names = config["layers_to_hook"]
    k = int(len(layer_names) * fraction)
    to_prune = random.sample(layer_names, k)

    removed = []
    for lname in to_prune:
        lid = lname.split(".")[-1]
        setattr(model.backbone.features, lid, nn.Identity())
        removed.append(lname)

    return model, removed


def taylor_prune(model, config, dataloader):
    """
    Placeholder for Taylor pruning strategy.
    """
    print("⚠️ Taylor pruning not implemented yet.")
    return model, []
