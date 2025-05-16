# scripts/map_voc.py

from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch

def evaluate_map_voc(model, dataloader, device="cuda", num_batches=50):
    """
    Evaluate mean Average Precision (mAP@0.5) on VOC dataset using torchmetrics.
    This is an approximation — use COCO tools for full compliance.
    """
    model.eval()
    model.to(device)
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for i, (img, target) in enumerate(dataloader):
            if i >= num_batches:
                break

            image = img[0].to(device)
            preds = model([image])[0]

            preds_formatted = [{
                "boxes": preds["boxes"].detach().cpu(),
                "scores": preds["scores"].detach().cpu(),
                "labels": preds["labels"].detach().cpu()
            }]

            target_dict = target["annotation"]
            objects = target_dict["object"]
            if not isinstance(objects, list):
                objects = [objects]

            true_boxes = []
            true_labels = []
            for obj in objects:
                box = obj["bndbox"]
                xmin = int(box["xmin"])
                ymin = int(box["ymin"])
                xmax = int(box["xmax"])
                ymax = int(box["ymax"])
                true_boxes.append([xmin, ymin, xmax, ymax])
                true_labels.append(1)  # Placeholder label

            targets_formatted = [{
                "boxes": torch.tensor(true_boxes).float(),
                "labels": torch.tensor(true_labels)
            }]

            metric.update(preds_formatted, targets_formatted)

    score = metric.compute()
    return round(score["map"].item(), 4)
