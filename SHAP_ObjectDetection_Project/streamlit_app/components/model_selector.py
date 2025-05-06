# streamlit_app/components/model_selector.py

import streamlit as st
import yaml
import os

def load_available_models(config_path="config"):
    models = []
    for fname in os.listdir(config_path):
        if fname.endswith(".yaml"):
            models.append(fname.replace(".yaml", ""))
    return sorted(models)

def select_model(config_path="config"):
    st.sidebar.header("🔍 Model Selection")
    models = load_available_models(config_path)
    selected_model = st.sidebar.selectbox("Choose a model", models)

    config_file = os.path.join(config_path, selected_model + ".yaml")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    st.sidebar.success(f"Loaded config for {selected_model}")
    return selected_model, config
