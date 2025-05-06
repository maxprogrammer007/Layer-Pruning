# streamlit_app/components/exporter.py

import streamlit as st
import pandas as pd
import os
import json

def export_ui():
    st.header("📤 Export Logs")

    options = st.radio("Select log type to export:", ["raw", "processed"])

    log_dir = f"logs/{options}"
    logs = []
    for fname in os.listdir(log_dir):
        if fname.endswith(".json"):
            with open(os.path.join(log_dir, fname), "r") as f:
                data = json.load(f)
                logs.append(data)

    df = pd.DataFrame(logs)
    st.dataframe(df)

    # Export as CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, f"{options}_logs.csv", "text/csv")

    # Export as LaTeX table
    latex = df.to_latex(index=False)
    st.download_button("⬇️ Download LaTeX", latex, f"{options}_logs.tex", "text/latex")
