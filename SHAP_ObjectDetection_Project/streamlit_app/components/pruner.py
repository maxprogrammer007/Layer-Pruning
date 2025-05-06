# streamlit_app/components/pruner.py

import streamlit as st

def prune_ui(shap_scores):
    st.header("✂️ Layer Pruning")

    if shap_scores is None:
        st.warning("Please run SHAP analysis first.")
        return None, None

    threshold = st.slider("Prune layers with SHAP value below:", 0.0, 1.0, 0.2, 0.01)
    auto_pruned = [layer for layer, score in shap_scores.items() if score < threshold]

    st.subheader("Auto-selected layers for pruning:")
    st.write(auto_pruned)

    manual = st.multiselect("Or manually select layers to prune:", list(shap_scores.keys()), default=auto_pruned)

    if st.button("Prune Selected Layers"):
        st.success(f"Will prune {len(manual)} layers.")
        return manual, threshold
    else:
        return None, threshold
