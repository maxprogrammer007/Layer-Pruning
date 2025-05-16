import os

def create_project_structure(base_dir="SHAP_Pruning_Research"):
    folders = [
        "config",
        "data",
        "logs/raw",
        "logs/processed",
        "models",
        "results/plots",
        "results/tables",
        "results/shap_scores",
        "scripts"
    ]

    files = [
        "main.py",
        "batch_run.py",
        "requirements.txt",
        "README.md",
        "scripts/__init__.py"
    ]

    # Create folders
    for folder in folders:
        path = os.path.join(base_dir, folder)
        os.makedirs(path, exist_ok=True)
        print(f"📁 Created folder: {path}")

    # Create empty placeholder files
    for file in files:
        path = os.path.join(base_dir, file)
        with open(path, "w") as f:
            pass
        print(f"📄 Created file: {path}")

    print("\n✅ Project structure initialized successfully.")

if __name__ == "__main__":
    create_project_structure()
