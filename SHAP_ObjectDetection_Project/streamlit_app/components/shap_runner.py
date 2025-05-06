# streamlit_app/components/shap_runner.py

import streamlit as st
import sys
import os
import torch
from scripts.shap_analysis import run_shap_analysis  # You will create this
import matplotlib.pyplot as plt

# Add scripts/ to path
# ✅ Add scripts/ to Python path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)
    
def shap_analysis_ui(model, config, dataloader):
    st.header("🧠 SHAP Layer-wise Analysis")

    if st.button("Run SHAP Analysis"):
        shap_scores = run_shap_analysis(model, config, dataloader)
        st.success("SHAP analysis completed.")

        # Bar plot of SHAP scores
        fig, ax = plt.subplots()
        layers = list(shap_scores.keys())
        scores = [shap_scores[l] for l in layers]
        ax.barh(layers, scores)
        ax.set_xlabel("Mean SHAP Value")
        ax.set_title("Layer-wise SHAP Contribution")
        st.pyplot(fig)

        return shap_scores
    else:
        st.info("Click the button to compute SHAP scores.")
        return None
