# streamlit_app/app.py

import streamlit as st
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection

from components.model_selector import select_model
from components.shap_runner import shap_analysis_ui
from components.pruner import prune_ui
from components.logger import save_log
from components.plot_viewer import plot_results_ui
from components.exporter import export_ui
from components.shap_runner import shap_analysis_ui


from utils.config_loader import load_config
from utils.evaluator import count_parameters, benchmark_fps
from scripts.map_voc import evaluate_map_voc
from scripts.prune import prune_layers
from scripts.shap_analysis import run_shap_analysis

import matplotlib.pyplot as plt


st.set_page_config(page_title="SHAP Pruning Dashboard", layout="wide")
st.title("🔬 SHAP-Based Object Detection Pruning Dashboard")

# === Step 1: Model Selection ===
model_name, config = select_model()
st.write(f"**Selected model:** `{model_name}`")

# === Step 2: Load Dataset ===
st.sidebar.header("📁 Dataset Loader")
transform = transforms.Compose([
    transforms.Resize(config["input_size"]),
    transforms.ToTensor()
])
dataset = VOCDetection(root="./data", year='2007', image_set='val', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
st.sidebar.success(f"Loaded {len(dataset)} validation images.")

# === Step 3: Load Model ===
st.sidebar.header("📦 Load Model")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
if config["model_name"] == "ssdlite320_mobilenet_v3_large":
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    model = ssdlite320_mobilenet_v3_large(pretrained=True)
elif config["model_name"] == "fasterrcnn_resnet50_fpn":
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(pretrained=True)
else:
    st.error("Unsupported model in config.")
    st.stop()

model.eval().to(device)
st.success(f"Model `{config['model_name']}` loaded on `{device}`")

# === Step 4: Evaluate Baseline ===
st.header("📊 Baseline Evaluation")
col1, col2 = st.columns(2)

with col1:
    fps = benchmark_fps(model, dataloader, device)
    st.metric("FPS (Before Pruning)", f"{fps:.2f}")

with col2:
    map_val = evaluate_map_voc(model, dataloader, device)
    st.metric("mAP@0.5 (Before)", f"{map_val:.3f}")

params = count_parameters(model)

# === Step 5: SHAP Analysis ===
shap_scores = shap_analysis_ui(model, config, dataloader)

if shap_scores:
    # Save RAW Log
    raw_log = {
        "model": model_name,
        "params": params,
        "fps": fps,
        "map": map_val,
        "layers": len(config["layers_to_hook"]),
        "shap_scores": shap_scores
    }
    save_log(raw_log, model_name, processed=False)

    # === Step 6: Pruning UI ===
    pruned_layers, threshold = prune_ui(shap_scores)
    if pruned_layers:
        pruned_model, removed_layers = prune_layers(model, config, pruned_layers)

        # === Step 7: Post-Pruning Evaluation ===
        st.header("🔁 Evaluation After Pruning")
        col1, col2 = st.columns(2)

        with col1:
            fps_after = benchmark_fps(pruned_model, dataloader, device)
            st.metric("FPS (After)", f"{fps_after:.2f}")

        with col2:
            map_after = evaluate_map_voc(pruned_model, dataloader, device)
            st.metric("mAP@0.5 (After)", f"{map_after:.3f}")

        # Save Processed Log
        processed_log = {
            "model": model_name,
            "fps_after": fps_after,
            "map_after": map_after,
            "layers_removed": len(removed_layers),
            "pruned_layers": removed_layers,
            "threshold": threshold
        }
        save_log(processed_log, model_name, processed=True)

        # Show tradeoff chart
        st.subheader("📉 Tradeoff Plot")
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar(["Before", "After"], [map_val, map_after], label="mAP", color="skyblue")
        ax2.plot(["Before", "After"], [fps, fps_after], "ro-", label="FPS")
        ax1.set_ylabel("mAP@0.5")
        ax2.set_ylabel("FPS")
        st.pyplot(fig)

# === Step 8: Export Logs & Tables ===
export_ui()
