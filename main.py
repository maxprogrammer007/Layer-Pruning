# main.py

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms

from scripts.model_loader import load_model
from scripts.evaluate import evaluate_model
from scripts.logger import save_log
from scripts.plotter import plot_shap_scores, plot_map_vs_fps
from scripts.shap_pruning import compute_shap_scores, prune_model_by_shap
from scripts.pruning_methods import l1_norm_prune, random_prune
from scripts.config_loader import load_config


def load_dataloader(input_size):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    dataset = VOCDetection(root="data", year='2007', image_set='val', download=True, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def run_pipeline(model_name, method="shap", threshold=0.2):
    print(f"\n🚀 Running pipeline for: {model_name} with method: {method.upper()}\n")

    config = load_config(model_name)
    model, device = load_model(model_name)
    dataloader = load_dataloader(config["input_size"])

    # === Evaluate Baseline ===
    print("📊 Evaluating baseline...")
    baseline = evaluate_model(model, dataloader, device)
    baseline["method"] = "baseline"
    save_log(baseline, model_name + "_baseline", processed=True)

    # === SHAP Pruning ===
    if method == "shap":
        print("\n🔍 Running SHAP-based pruning...")
        shap_scores = compute_shap_scores(model, dataloader, config, device=device)
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

        save_log(after, model_name + "_shap", processed=True)
        plot_shap_scores(shap_scores, model_name)
        plot_map_vs_fps(baseline, after, model_name)


    elif method == "l1":
        print("\n📉 Running L1-norm pruning...")
        model, pruned_layers = l1_norm_prune(model, config, dataloader, threshold, device)
        after = evaluate_model(model, dataloader, device)
        after.update({
            "method": "l1",
            "layers_removed": len(pruned_layers),
            "pruned_layers": pruned_layers
        })
        save_log(after, model_name + "_l1", processed=True)
        plot_map_vs_fps(baseline, after, model_name)

    elif method == "random":
        print("\n🎲 Running random pruning...")
        model, pruned_layers = random_prune(model, config, fraction=threshold)
        after_eval = evaluate_model(model, dataloader, device)
        after = {
            "method": "shap",
            "map_after": after_eval["map"],
             "fps_after": after_eval["fps"],
            "params_after": after_eval["params"],
            "flops_after": after_eval["flops"],
            "layers_removed": len(pruned_layers),
            "pruned_layers": pruned_layers
        }

        save_log(after, model_name + "_random", processed=True)
        plot_map_vs_fps(baseline, after, model_name)

    else:
        raise ValueError(f"❌ Unsupported pruning method: {method}")

    print("\n✅ Pipeline complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP-based or comparative pruning pipeline")
    parser.add_argument("--model", required=True, help="Model name from config folder (e.g., mobilenet_ssd)")
    parser.add_argument("--method", default="shap", choices=["shap", "l1", "random"], help="Pruning method")
    parser.add_argument("--threshold", type=float, default=0.2, help="SHAP or L1 threshold, or random fraction")
    args = parser.parse_args()

    run_pipeline(model_name=args.model, method=args.method, threshold=args.threshold)
