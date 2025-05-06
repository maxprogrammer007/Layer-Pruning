# streamlit_app/components/plot_viewer.py

import streamlit as st
import matplotlib.pyplot as plt
import json
import os

def plot_results_ui(log_path):
    st.header("📊 Results Visualizer")

    if not os.path.exists(log_path):
        st.warning("No logs found.")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)

    # Plot FPS vs mAP
    st.subheader("Trade-off: Accuracy vs Speed")
    fig, ax = plt.subplots()
    ax.bar(["Before", "After"], [data["map"], data.get("map_after", 0)], label="mAP")
    ax.set_ylabel("mAP@0.5")
    ax2 = ax.twinx()
    ax2.plot(["Before", "After"], [data["fps"], data.get("fps_after", 0)], "r-o", label="FPS")
    ax2.set_ylabel("FPS")

    st.pyplot(fig)
