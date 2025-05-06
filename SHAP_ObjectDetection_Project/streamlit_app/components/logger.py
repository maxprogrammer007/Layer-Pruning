# streamlit_app/components/logger.py

import json
import os

def save_log(data, model_name, processed=False):
    log_type = "processed" if processed else "raw"
    log_dir = f"logs/{log_type}"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, model_name + ".json")

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {log_type} log to {path}")
