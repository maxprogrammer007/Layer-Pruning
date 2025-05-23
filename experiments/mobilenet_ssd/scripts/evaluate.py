# experiments/mobilenet_ssd/scripts/evaluate.py
import time, json
import torch
from scripts.model_loader import load_model
from scripts.map_voc      import compute_map
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from ptflops import get_model_complexity_info

def run_evaluation(cfg, out_json):
    device = torch.device(cfg['device'])
    model  = load_model(cfg['model'], device)

    ds = VOCDetection(
        cfg['dataset']['root'],
        year='2007',
        image_set=cfg['dataset']['split'],
        transform=ToTensor()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    # FPS
    N = 100
    start = time.time()
    for i, (img, _) in enumerate(loader):
        if i >= N: break
        with torch.no_grad():
            _ = model(img.to(device))
    fps = N / (time.time() - start)

    # mAP
    mAP = compute_map(model, ds, device)

    # FLOPs & params
    flops, params = get_model_complexity_info(
        model, (3,300,300),
        as_strings=False, print_per_layer_stat=False
    )

    results = {'mAP': mAP, 'FPS': fps, 'FLOPs': flops, 'Params': params}
    with open(out_json,'w') as f:
        json.dump(results, f, indent=2)
    print(f"[evaluate] saved results to {out_json}")
