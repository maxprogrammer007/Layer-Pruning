# scripts/logger.py

import os
import json

def save_log(data, model_name, processed=False):
    """
    Save metrics and SHAP/pruning info as a JSON log.
    """
    log_type = "processed" if processed else "raw"
    log_dir = f"logs/{log_type}"
    os.makedirs(log_dir, exist_ok=True)

    filename = f"{model_name}.json"
    path = os.path.join(log_dir, filename)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"📁 Saved {log_type} log: {path}")


def load_logs(processed=True):
    """
    Load all JSON logs from the specified log type.
    Returns a list of dictionaries.
    """
    log_type = "processed" if processed else "raw"
    log_dir = f"logs/{log_type}"

    logs = []
    for fname in os.listdir(log_dir):
        if fname.endswith(".json"):
            with open(os.path.join(log_dir, fname), "r") as f:
                logs.append(json.load(f))

    return logs
