# scripts/config_loader.py

import os
import yaml

def load_config(model_name, config_dir="config"):
    """
    Loads a YAML config for the given model.
    """
    config_path = os.path.join(config_dir, f"{model_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config not found for: {model_name}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def list_configs(config_dir="config"):
    """
    List available configs in the config folder.
    """
    return [f.replace(".yaml", "") for f in os.listdir(config_dir) if f.endswith(".yaml")]
