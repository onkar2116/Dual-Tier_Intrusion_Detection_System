import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time


def render_metric_row():
    """Render top-row KPI metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Traffic", "12,450", delta="↑ 230")
    col2.metric("Attacks Detected", "187", delta="↑ 12")
    col3.metric("Adversarial Attacks", "23", delta="↑ 3")
    col4.metric("False Positive Rate", "3.2%", delta="-0.5%")
    col5.metric("System Uptime", "99.8%")


def render_alert_feed():
    """Render scrollable alert feed."""
    alerts = [
        {"time": "14:32:05", "type": "SYN Flood", "tier": 1, "severity": "HIGH"},
        {"time": "14:31:58", "type": "Adversarial Evasion", "tier": 3, "severity": "CRITICAL"},
        {"time": "14:31:45", "type": "Port Scan", "tier": 2, "severity": "MEDIUM"},
        {"time": "14:31:30", "type": "SQL Injection", "tier": 1, "severity": "CRITICAL"},
        {"time": "14:31:12", "type": "DoS Attack", "tier": 2, "severity": "HIGH"},
        {"time": "14:30:55", "type": "Brute Force", "tier": 1, "severity": "HIGH"},
        {"time": "14:30:40", "type": "Adversarial PGD", "tier": 3, "severity": "CRITICAL"},
    ]

    for alert in alerts:
        severity_color = {
            "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
        }.get(alert["severity"], "⚪")

        st.markdown(
            f"{severity_color} **{alert['time']}** | Tier {alert['tier']} | "
            f"{alert['type']} ({alert['severity']})"
        )


def render_tier_breakdown():
    """Render tier-wise detection breakdown."""
    st.subheader("⚙️ Tier-wise Detection Breakdown")
    tier_col1, tier_col2, tier_col3 = st.columns(3)

    with tier_col1:
        st.markdown("### Tier 1: Signature")
        st.metric("Detections", "45")
        st.metric("Avg Speed", "0.2ms")
        st.progress(0.95, text="Detection Rate: 95%")

    with tier_col2:
        st.markdown("### Tier 2: ML")
        st.metric("Detections", "119")
        st.metric("Avg Speed", "12ms")
        st.progress(0.92, text="Detection Rate: 92%")

    with tier_col3:
        st.markdown("### Tier 3: Adversarial")
        st.metric("Detections", "23")
        st.metric("Avg Speed", "35ms")
        st.progress(0.85, text="Detection Rate: 85%")


def render_attack_simulation():
    """Render attack simulation interface."""
    st.subheader("🎯 Adversarial Attack Simulator")

    attack_type = st.selectbox("Attack Type", ["FGSM", "PGD", "C&W", "DeepFool"])
    epsilon = st.slider("Perturbation Epsilon", 0.01, 0.5, 0.1, 0.01)
    n_samples = st.number_input("Number of Samples", min_value=10, max_value=1000, value=100)

    if st.button("Launch Simulation"):
        with st.spinner(f"Generating {attack_type} adversarial samples..."):
            time.sleep(2)

            # Simulated results
            success_rate = np.random.uniform(50, 90)
            detected_rate = np.random.uniform(70, 95)

            st.success(f"Generated {n_samples} {attack_type} adversarial samples (epsilon={epsilon})")

            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "attack_type": attack_type,
                    "epsilon": epsilon,
                    "samples_generated": n_samples,
                    "attack_success_rate": f"{success_rate:.1f}%",
                    "detected_by_tier3": f"{detected_rate:.1f}%"
                })
            with col2:
                fig = go.Figure(data=[
                    go.Bar(name='Attack Success', x=[attack_type], y=[success_rate], marker_color='red'),
                    go.Bar(name='Detected by Tier 3', x=[attack_type], y=[detected_rate], marker_color='green'),
                ])
                fig.update_layout(title='Attack vs Detection Rate', barmode='group')
                st.plotly_chart(fig, use_container_width=True)


def render_model_performance():
    """Render model performance comparison."""
    st.subheader("📈 Model Performance Comparison")

    performance_data = {
        'System': ['Signature Only', 'ML Only (Non-Robust)', 'Dual-Tier (Non-Robust)', 'Our Three-Tier Robust'],
        'Accuracy': [85.2, 93.5, 94.8, 95.2],
        'FPR': [2.0, 5.0, 4.0, 3.5],
        'Detection Rate': [78, 92, 93, 94],
        'Robust Accuracy': [0, 45, 48, 88],
        'Adv Detection': [0, 0, 0, 85]
    }
    df_perf = pd.DataFrame(performance_data)
    st.dataframe(df_perf, use_container_width=True)

    fig = go.Figure(data=[
        go.Bar(name='Clean Accuracy', x=performance_data['System'], y=performance_data['Accuracy']),
        go.Bar(name='Robust Accuracy', x=performance_data['System'], y=performance_data['Robust Accuracy'])
    ])
    fig.update_layout(barmode='group', title='Clean vs Robust Accuracy Comparison')
    st.plotly_chart(fig, use_container_width=True)
