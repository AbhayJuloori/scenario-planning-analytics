import os
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

st.set_page_config(page_title="Urban Demand Planning", layout="wide")

st.title("Urban Demand Planning & Scenario Analytics")

metrics_path = os.path.join(REPORTS_DIR, "model_metrics.csv")
scenarios_path = os.path.join(REPORTS_DIR, "scenario_results.csv")

if os.path.exists(metrics_path):
    st.subheader("Model Metrics")
    metrics_df = pd.read_csv(metrics_path)
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.warning("Run `python src/train_model.py` to generate model metrics.")

if os.path.exists(scenarios_path):
    st.subheader("Scenario Planning Results")
    scenario_df = pd.read_csv(scenarios_path)

    fig = px.line(
        scenario_df,
        x="date",
        y="service_level",
        color="scenario",
        title="Service Level Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.line(
        scenario_df,
        x="date",
        y="overtime_cost_index",
        color="scenario",
        title="Overtime Cost Index",
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Run `python src/score_scenarios.py` to generate scenario results.")
