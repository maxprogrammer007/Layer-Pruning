# scripts/plotter.py

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_shap_scores(shap_scores, model_name, save_dir="results/plots"):
    """
    Bar plot of SHAP scores per layer.
    """
    os.makedirs(save_dir, exist_ok=True)
    keys = list(shap_scores.keys())
    values = [shap_scores[k] for k in keys]

    plt.figure(figsize=(10, 6))
    plt.barh(keys, values, color='skyblue')
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title(f"SHAP Layer-Wise Scores - {model_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_shap.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ SHAP plot saved to {path}")


def plot_map_vs_fps(before_metrics, after_metrics, model_name, save_dir="results/plots"):
    """
    Trade-off plot: mAP vs FPS (before vs after pruning)
    """
    os.makedirs(save_dir, exist_ok=True)
    labels = ["Before", "After"]
    mAPs = [before_metrics["map"], after_metrics["map_after"]]
    FPSs = [before_metrics["fps"], after_metrics["fps_after"]]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.bar(labels, mAPs, color='blue', alpha=0.6, label="mAP")
    ax2.plot(labels, FPSs, color='red', marker='o', label="FPS")

    ax1.set_ylabel("mAP@0.5")
    ax2.set_ylabel("FPS")
    plt.title(f"Accuracy vs Speed - {model_name}")
    plt.tight_layout()

    path = os.path.join(save_dir, f"{model_name}_tradeoff.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Trade-off plot saved to {path}")


def plot_pruning_comparison(logs, model_name="comparison", save_dir="results/plots"):
    """
    Plot mAP and FPS for different pruning methods (SHAP, L1, Random).
    """
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(logs)

    fig, ax1 = plt.subplots()
    methods = df["method"]
    mAPs = df["map_after"]
    FPSs = df["fps_after"]

    ax1.bar(methods, mAPs, color='green', alpha=0.6, label="mAP")
    ax2 = ax1.twinx()
    ax2.plot(methods, FPSs, color='orange', marker='o', label="FPS")

    ax1.set_ylabel("mAP@0.5")
    ax2.set_ylabel("FPS")
    plt.title("Comparison of Pruning Strategies")
    plt.tight_layout()

    path = os.path.join(save_dir, f"{model_name}_method_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Pruning comparison plot saved to {path}")
