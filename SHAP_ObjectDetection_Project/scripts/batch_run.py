import os
import torch
import json
from utils.config_loader import list_configs, load_config
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms
from scripts.shap_analysis import run_shap_analysis
from scripts.evaluator import benchmark_fps, count_parameters
from scripts.logger import save_log
from scripts.map_voc import evaluate_map_voc

def run_all_models():
    configs = list_configs()
    for model_name in configs:
        config = load_config(model_name)
        print(f"\n📦 Running {model_name}...")

        model = torch.hub.load('pytorch/vision:v0.10.0', config["model_name"], pretrained=True)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(config["input_size"]),
            transforms.ToTensor()
        ])
        dataset = VOCDetection(root="./data", year='2007', image_set='val', download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Evaluate baseline metrics
        fps = benchmark_fps(model, dataloader, device="cuda")
        map_val = evaluate_map_voc(model, dataloader, device="cuda")
        params = count_parameters(model)
        shap_scores = run_shap_analysis(model, config, dataloader)

        log_data = {
            "model": model_name,
            "params": params,
            "fps": fps,
            "map": map_val,
            "layers": len(config["layers_to_hook"]),
            "shap_scores": shap_scores
        }

        save_log(log_data, model_name, processed=False)

if __name__ == "__main__":
    run_all_models()
