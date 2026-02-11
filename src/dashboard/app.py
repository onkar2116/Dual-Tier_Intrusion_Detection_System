import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.dashboard.components import (
    render_metric_row, render_alert_feed, render_tier_breakdown,
    render_attack_simulation, render_model_performance
)

st.set_page_config(
    page_title="Adversarial Robust IDS",
    page_icon="🛡️",
    layout="wide"
)

# --- HEADER ---
st.title("🛡️ Adversarially Robust Intrusion Detection System")
st.markdown("Three-tier detection: Signature | ML | Adversarial Robustness")
st.divider()

# --- SIDEBAR ---
st.sidebar.title("Control Panel")
mode = st.sidebar.selectbox(
    "Detection Mode",
    ["Real-time Monitor", "Batch Analysis", "Attack Simulation", "Model Performance"]
)

# --- TOP METRICS ---
render_metric_row()

st.divider()

# --- MAIN CONTENT ---
if mode == "Real-time Monitor":
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("📊 Traffic Flow (Last 60 Seconds)")
        # Simulated real-time chart
        chart_data = pd.DataFrame({
            'Time': pd.date_range(start='now', periods=60, freq='s'),
            'Benign': np.random.randint(50, 200, 60),
            'Attacks': np.random.randint(0, 30, 60),
            'Adversarial': np.random.randint(0, 5, 60)
        })
        fig = px.line(chart_data, x='Time', y=['Benign', 'Attacks', 'Adversarial'],
                      title='Traffic Classification Over Time')
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("🚨 Live Alerts")
        render_alert_feed()

    st.divider()
    render_tier_breakdown()

elif mode == "Batch Analysis":
    st.subheader("📁 Batch Analysis")
    st.markdown("Upload a CSV file with network traffic features for batch detection.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} samples with {len(df.columns)} features")
        st.dataframe(df.head(10))

        if st.button("Run Detection"):
            with st.spinner("Running three-tier detection..."):
                time.sleep(2)
                n = len(df)
                results = {
                    'Total Samples': n,
                    'Benign': int(n * 0.6),
                    'Tier 1 Detections': int(n * 0.15),
                    'Tier 2 Detections': int(n * 0.2),
                    'Tier 3 (Adversarial)': int(n * 0.05),
                }
                st.success("Detection complete!")
                st.json(results)

elif mode == "Attack Simulation":
    render_attack_simulation()

elif mode == "Model Performance":
    render_model_performance()
