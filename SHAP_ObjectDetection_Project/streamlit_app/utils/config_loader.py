# streamlit_app/utils/config_loader.py

import os
import yaml

def list_configs(config_dir="config"):
    """Lists available YAML config files."""
    return [f.replace(".yaml", "") for f in os.listdir(config_dir) if f.endswith(".yaml")]

def load_config(model_name, config_dir="config"):
    """Loads YAML config for a specific model."""
    path = os.path.join(config_dir, model_name + ".yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found for model: {model_name}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)
