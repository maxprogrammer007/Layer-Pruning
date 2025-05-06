# streamlit_app/utils/evaluator.py

import time
import torch
from torchinfo import summary

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_fps(model, dataloader, device, num_batches=10):
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

    return num_batches / total_time

# streamlit_app/utils/evaluator.py

def dummy_map(model, dataloader):
    # Replace this with real VOC mAP calculator if needed
    import random
    return round(0.70 + random.uniform(-0.05, 0.05), 3)
