# scripts/map_voc.py

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

def evaluate_map_voc(model, dataloader, device="cuda", num_batches=100):
    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, (img, target) in enumerate(dataloader):
            if i >= num_batches:
                break

            img = img.to(device)
            preds = model([img[0]])

            # Format predictions + targets to torchmetrics format
            preds_formatted = [{
                "boxes": preds[0]["boxes"].cpu(),
                "scores": preds[0]["scores"].cpu(),
                "labels": preds[0]["labels"].cpu()
            }]
            targets_formatted = [{
                "boxes": target["annotation"]["object"]["bndbox"],
                "labels": torch.tensor([int(target["annotation"]["object"]["name"] == 'person')])
            }]

            # NOTE: For multiple objects, you'll need to parse all properly

            metric.update(preds_formatted, targets_formatted)

    results = metric.compute()
    return round(results["map"].item(), 4)
