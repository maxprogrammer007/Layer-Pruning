# experiments/mobilenet_ssd/scripts/shap_pruning.py
import json, random, torch, shap
from scripts.model_loader import load_model
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor

def compute_shap_scores(cfg):
    device = torch.device(cfg['device'])
    model  = load_model(cfg['model'], device)

    ds = VOCDetection(
        cfg['dataset']['root'], year='2007',
        image_set=cfg['dataset']['split'], transform=ToTensor()
    )

    bkg_idx    = random.sample(range(len(ds)), cfg['shap']['background_size'])
    background = torch.stack([ds[i][0] for i in bkg_idx]).to(device)
    explainer  = shap.GradientExplainer(model, background)

    batch_idx = random.sample(range(len(ds)), cfg['shap']['batch_size'])
    batch     = torch.stack([ds[i][0] for i in batch_idx]).to(device)
    shap_vals = explainer.shap_values(batch)

    shap_scores = {
        layer: float(torch.abs(torch.tensor(val)).mean().item())
        for layer, val in zip(model.layer_names, shap_vals)
    }

    out_path = f"{cfg['logging']['log_dir']}/shap_scores.json"
    with open(out_path,'w') as f:
        json.dump(shap_scores, f, indent=2)
    print(f"[shap_pruning] saved SHAP scores to {out_path}")
    return shap_scores

def apply_prune(cfg, scores, method="shap"):
    threshold = cfg['prune']['threshold']
    return {layer: (score >= threshold) for layer, score in scores.items()}
