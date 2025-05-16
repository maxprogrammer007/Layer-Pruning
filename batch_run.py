# batch_run.py

import os
import traceback
from scripts.config_loader import list_configs, load_config
from scripts.model_loader import load_model
from scripts.evaluate import evaluate_model
from scripts.shap_pruning import compute_shap_scores, prune_model_by_shap
from scripts.pruning_methods import l1_norm_prune, random_prune
from scripts.logger import save_log
from scripts.plotter import plot_shap_scores, plot_map_vs_fps
from scripts.map_voc import evaluate_map_voc
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms


def load_dataloader(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    return DataLoader(VOCDetection(root="data", year='2007', image_set='val', download=True, transform=transform),
                      batch_size=1, shuffle=True)


def run_one(model_name, method, threshold):
    print(f"\n🔁 Model: {model_name.upper()} | Method: {method.upper()} | Threshold: {threshold}")

    try:
        # Load model, config, data
        config = load_config(model_name)
        model, device = load_model(model_name)
        dataloader = load_dataloader(config["input_size"])

        # Evaluate baseline
        baseline = evaluate_model(model, dataloader, device)
        baseline["method"] = "baseline"
        save_log(baseline, f"{model_name}_baseline", processed=True)

        # Prune and evaluate
        if method == "shap":
            shap_scores = compute_shap_scores(model, dataloader, config, device)
            model_pruned, pruned_layers = prune_model_by_shap(model, shap_scores, threshold=threshold)
            after_eval = evaluate_model(model_pruned, dataloader, device)
            after = {
                "method": "shap",
                "model": model_name,
                "map_after": after_eval["map"],
                "fps_after": after_eval["fps"],
                "params_after": after_eval["params"],
                "flops_after": after_eval["flops"],
                "layers_removed": len(pruned_layers),
                "pruned_layers": pruned_layers
            }
            save_log(after, f"{model_name}_shap", processed=True)
            plot_shap_scores(shap_scores, model_name)
            plot_map_vs_fps(baseline, after, model_name)

        elif method == "l1":
            model_pruned, pruned_layers = l1_norm_prune(model, config, dataloader, threshold, device)
            after_eval = evaluate_model(model_pruned, dataloader, device)
            after = {
                "method": "l1",
                "model": model_name,
                "map_after": after_eval["map"],
                "fps_after": after_eval["fps"],
                "params_after": after_eval["params"],
                "flops_after": after_eval["flops"],
                "layers_removed": len(pruned_layers),
                "pruned_layers": pruned_layers
            }
            save_log(after, f"{model_name}_l1", processed=True)
            plot_map_vs_fps(baseline, after, model_name)

        elif method == "random":
            model_pruned, pruned_layers = random_prune(model, config, fraction=threshold)
            after_eval = evaluate_model(model_pruned, dataloader, device)
            after = {
                "method": "random",
                "model": model_name,
                "map_after": after_eval["map"],
                "fps_after": after_eval["fps"],
                "params_after": after_eval["params"],
                "flops_after": after_eval["flops"],
                "layers_removed": len(pruned_layers),
                "pruned_layers": pruned_layers
            }
            save_log(after, f"{model_name}_random", processed=True)
            plot_map_vs_fps(baseline, after, model_name)

        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"✅ Completed: {model_name} | {method}")

    except Exception as e:
        print(f"❌ Failed: {model_name} | {method}")
        print(traceback.format_exc())


def run_all(models=None, methods=None, threshold=0.2):
    if models is None:
        models = list_configs()

    if methods is None:
        methods = ["shap", "l1", "random"]

    print(f"📊 Starting batch run on {len(models)} models with methods: {methods}")

    for model_name in models:
        for method in methods:
            run_one(model_name, method, threshold)


if __name__ == "__main__":
    run_all()
