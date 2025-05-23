# experiments/mobilenet_ssd/scripts/l1_pruning.py
import torch
from scripts.model_loader import load_model

def compute_l1_scores(cfg):
    device = torch.device(cfg['device'])
    model  = load_model(cfg['model'], device)
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            scores[name] = float(torch.mean(torch.abs(module.weight)).item())
    return scores

def apply_prune(cfg, scores, method="l1"):
    threshold = cfg['prune']['threshold']
    return {layer: (score >= threshold) for layer, score in scores.items()}
