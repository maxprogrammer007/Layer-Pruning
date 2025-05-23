# experiments/mobilenet_ssd/scripts/config_loader.py
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
