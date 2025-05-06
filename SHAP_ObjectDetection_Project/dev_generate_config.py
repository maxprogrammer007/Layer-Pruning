# dev_generate_config.py

import torch
import yaml
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

model = ssdlite320_mobilenet_v3_large(pretrained=True)
layer_names = [f"features.{name}" for name in dict(model.backbone.features.named_children()).keys()]

config = {
    "model_name": "ssdlite320_mobilenet_v3_large",
    "input_size": [320, 320],
    "layers_to_hook": layer_names,
    "baseline_fps": None,
    "baseline_map": None
}

with open("config/mobilenet_ssd.yaml", "w") as f:
    yaml.dump(config, f)

print("✅ YAML generated.")
