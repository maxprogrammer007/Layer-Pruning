# main.py

import argparse
import os
import torch
import json
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn

from scripts.shap_analysis import run_shap_analysis
from scripts.prune import prune_layers
from scripts.logger import save_log
from scripts.map_voc import evaluate_map_voc
from streamlit_app.utils.evaluator import benchmark_fps, count_parameters
from streamlit_app.utils.config_loader import load_config


def load_model(name):
    if name == "mobilenet_ssd":
        return ssdlite320_mobilenet_v3_large(pretrained=True)
    elif name == "resnet50_frcnn":
        return fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {name}")


def load_dataloader(input_size, batch_size=1):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    dataset = VOCDetection(root="./data", year='2007', image_set='val', download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def plot_shap(shap_scores, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(shap_scores.keys())
    values = [shap_scores[k] for k in layers]
    ax.barh(layers, values)
    ax.set_title("SHAP Layer-Wise Contribution")
    ax.set_xlabel("Mean Absolute SHAP Value")
    plot_path = f"results/plots/{model_name}_shap.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📊 SHAP bar plot saved to {plot_path}")


def plot_tradeoff(raw_log, processed_log, model_name):
    labels = ["Before", "After"]
    mAPs = [raw_log["map"], processed_log["map_after"]]
    FPSs = [raw_log["fps"], processed_log["fps_after"]]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(labels, mAPs, color='b', alpha=0.6, label='mAP')
    ax2.plot(labels, FPSs, color='r', marker='o', label='FPS')

    ax1.set_ylabel("mAP@0.5")
    ax2.set_ylabel("FPS")
    plt.title("Trade-off: Accuracy vs Speed")

    plot_path = f"results/plots/{model_name}_tradeoff.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📊 Trade-off plot saved to {plot_path}")


def main(model_name, threshold):
    print(f"🚀 Running full pipeline for {model_name}...")

    # Load model + config
    config = load_config(model_name)
    model = load_model(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load data
    dataloader = load_dataloader(config["input_size"])

    # Step 1: Raw Evaluation
    print("📈 Evaluating baseline...")
    fps = benchmark_fps(model, dataloader, device)
    map_val = evaluate_map_voc(model, dataloader, device)
    params = count_parameters(model)

    # Step 2: SHAP Analysis
    print("🧠 Running SHAP...")
    shap_scores = run_shap_analysis(model, config, dataloader)
    plot_shap(shap_scores, model_name)

    raw_log = {
        "model": model_name,
        "params": params,
        "fps": fps,
        "map": map_val,
        "layers": len(config["layers_to_hook"]),
        "shap_scores": shap_scores
    }
    save_log(raw_log, model_name, processed=False)

    # Step 3: Pruning
    print("✂️ Pruning based on SHAP < threshold =", threshold)
    to_prune = [k for k, v in shap_scores.items() if v < threshold]
    pruned_model, removed_layers = prune_layers(model, config, to_prune)

    # Step 4: Post-pruning Evaluation
    print("📉 Evaluating after pruning...")
    fps_after = benchmark_fps(pruned_model, dataloader, device)
    map_after = evaluate_map_voc(pruned_model, dataloader, device)

    processed_log = {
        "model": model_name,
        "fps_after": fps_after,
        "map_after": map_after,
        "layers_removed": len(removed_layers),
        "pruned_layers": removed_layers,
        "threshold": threshold
    }
    save_log(processed_log, model_name, processed=True)
    plot_tradeoff(raw_log, processed_log, model_name)

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP-based pruning pipeline")
    parser.add_argument('--model', required=True, help="Model name from config (e.g., mobilenet_ssd)")
    parser.add_argument('--threshold', type=float, default=0.2, help="SHAP threshold for pruning")
    args = parser.parse_args()

    main(args.model, args.threshold)
