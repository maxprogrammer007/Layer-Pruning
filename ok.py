import os

def create_project_structure(base_path="SHAP_ObjectDetection_Project"):
    folders = [
        "data",
        "models/mobilenet_ssd",
        "models/resnet_frcnn",
        "models/efficientdet",
        "models/detr",
        "models/yolov8",
        "models/resnext_frcnn",
        "logs/raw",
        "logs/processed",
        "results/shap_scores",
        "results/plots",
        "results/tables",
        "results/reports",
        "scripts",
        "notebooks",
        "config",
        "streamlit_app/components",
        "streamlit_app/utils"
    ]

    for folder in folders:
        full_path = os.path.join(base_path, folder)
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ Created: {full_path}")

if __name__ == "__main__":
    create_project_structure()
