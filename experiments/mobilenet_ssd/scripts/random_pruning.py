# experiments/mobilenet_ssd/scripts/random_pruning.py
import random
from scripts.model_loader import load_model

def apply_random_prune(cfg, ratio):
    model = load_model(cfg['model'], cfg['device'])
    layers = [
        name for name, m in model.named_modules()
        if isinstance(m, torch.nn.Conv2d)
    ]
    k    = int(len(layers) * ratio)
    drop = set(random.sample(layers, k))
    return {l: (l not in drop) for l in layers}
