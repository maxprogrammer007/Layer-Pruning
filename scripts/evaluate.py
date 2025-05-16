# scripts/evaluate.py

import time
import torch
from torchinfo import summary
from scripts.map_voc import evaluate_map_voc


def count_parameters(model):
    """
    Count trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_fps(model, dataloader, device="cuda", num_batches=20):
    """
    Estimate FPS (Frames Per Second) of the model on given dataloader.
    """
    model.eval()
    total_time = 0.0

    with torch.no_grad():
        for i, (img, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            img = img.to(device)
            start = time.time()
            _ = model([img[0]])
            end = time.time()
            total_time += (end - start)

    return round(num_batches / total_time, 2)


def compute_flops(model, input_size=(3, 320, 320)):
    """
    (Optional) Estimate FLOPs using torchinfo.
    """
    try:
        s = summary(model.backbone, input_size=(1, *input_size), verbose=0)
        return s.total_mult_adds / 1e6  # in MFLOPs
    except:
        return None


def evaluate_model(model, dataloader, device="cuda"):
    """
    Evaluate and return key metrics: Params, FPS, mAP, FLOPs
    """
    print("📈 Running model evaluation...")
    fps = benchmark_fps(model, dataloader, device)
    map_val = evaluate_map_voc(model, dataloader, device)
    params = count_parameters(model)
    flops = compute_flops(model)

    return {
        "params": params,
        "fps": fps,
        "map": map_val,
        "flops": flops
    }
