# ============================================================
#
#   PROJECT:     Predictive Aircraft Health Monitoring System
#                (PAHMS)
#
#   COURSE:      CIS-498 — Information Systems Capstone
#
#   TEAM MEMBERS:
#                Shazal    — Project Lead / Developer
#                Mary      — Data Analysis
#                Agrima    — Dashboard Design
#
#   SUBMISSION DATE: May 2026
#
#   DESCRIPTION:
#                PAHMS is a prototype aircraft health monitoring
#                dashboard built using Python and Streamlit.
#                It processes real NASA Aviation Safety Reporting
#                System (ASRS) data to classify aircraft health
#                status (Normal / Warning / Critical), generate
#                maintenance alerts, and display fleet analytics
#                through an interactive web dashboard.
#
#   TECHNOLOGY STACK:
#                Python 3.10+  — Core language
#                Pandas        — Data loading and processing
#                NumPy         — Numerical operations
#                Streamlit     — Web dashboard framework
#                Plotly        — Charts and visualizations
#
#   DATA SOURCE:
#                NASA Aviation Safety Reporting System (ASRS)
#                50,000 aviation incident records (2005–2023)
#
#   HOW TO RUN:
#                1. pip install -r requirements.txt
#                2. streamlit run app.py
#                3. Open http://localhost:8501 in your browser
#
#   ARCHITECTURE LAYERS:
#                Layer 1 — Data Input
#                Layer 2 — Data Processing
#                Layer 3 — Analytics & Alert Engine
#                Layer 4 — Dashboard (Streamlit UI)
#                Layer 5 — User Action (Maintenance Decisions)
#
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---- PAGE CONFIGURATION ----
st.set_page_config(
    page_title="PAHMS | Aircraft Health Monitoring",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS STYLING ----
# Makes the dashboard look clean and professional
st.markdown("""
    <style>
        /* Main background */
        .main { background-color: #0e1117; }

        /* Metric card styling */
        [data-testid="metric-container"] {
            background-color: #1c2333;
            border: 1px solid #2d3a52;
            border-radius: 10px;
            padding: 14px 18px;
        }

        /* Header bar */
        .pahms-header {
            background: linear-gradient(90deg, #1a3a6e 0%, #0d1b3e 100%);
            border-radius: 12px;
            padding: 18px 28px;
            margin-bottom: 20px;
            border-left: 5px solid #2e7bff;
        }

        /* Status badge styling */
        .badge-critical {
            background-color: #7f1d1d;
            color: #fca5a5;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-warning {
            background-color: #78350f;
            color: #fde68a;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-normal {
            background-color: #14532d;
            color: #86efac;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }

        /* Section divider */
        hr { border-color: #2d3a52; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }

        /* Alert box */
        .alert-box {
            background-color: #1f0000;
            border-left: 4px solid #ef4444;
            border-radius: 6px;
            padding: 10px 16px;
            margin-bottom: 8px;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================
# LAYER 1: DATA INPUT — Load and validate the dataset
# ============================================================

@st.cache_data  # Cache so the data doesn't reload every time a filter changes
def load_data():
    """
    Load the predictive maintenance CSV file.
    Returns a cleaned Pandas DataFrame ready for analysis.
    """
    # Build an absolute path so the file is found both locally and on Streamlit Cloud
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "predictive_maintenance.csv")
    df = pd.read_csv(data_path, low_memory=False)

    # Convert report_date to proper datetime format
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    # Ensure health_status is a category for faster filtering
    df["health_status"] = df["health_status"].astype("category")

    return df


# ============================================================
# LAYER 2: DATA PROCESSING — Clean and prepare records
# ============================================================

def filter_data(df, selected_models, selected_phases, selected_statuses, year_range):
    """
    Apply user-selected filters from the sidebar.
    Returns a filtered copy of the DataFrame.
    """
    filtered = df.copy()

    # Filter by aircraft model
    if selected_models:
        filtered = filtered[filtered["aircraft_model"].isin(selected_models)]

    # Filter by flight phase
    if selected_phases:
        filtered = filtered[filtered["flight_phase"].isin(selected_phases)]

    # Filter by health status
    if selected_statuses:
        filtered = filtered[filtered["health_status"].isin(selected_statuses)]

    # Filter by year range
    filtered = filtered[
        (filtered["report_year"] >= year_range[0]) &
        (filtered["report_year"] <= year_range[1])
    ]

    return filtered


# ============================================================
# LAYER 3: ANALYTICS & ALERT ENGINE — Classification + Alerts
# ============================================================

def reclassify_health(df):
    """
    Rule-based health classification logic.
    This is the core of the analytics engine.

    Rules:
    - CRITICAL: Equipment failure, fire/smoke anomalies
    - WARNING:  Malfunctioning components, aircraft root cause
    - NORMAL:   All other records
    """
    conditions = [
        df["anomaly_type"].str.contains("critical", case=False, na=False) |
        (df["component_status"].str.lower() == "failed") |
        df["anomaly_type"].str.contains("smoke|fire", case=False, na=False),

        df["component_status"].str.lower().str.contains("malfunction", na=False) |
        (df["root_cause"].str.lower() == "aircraft") |
        df["anomaly_type"].str.contains("equipment problem", case=False, na=False),
    ]
    choices = ["Critical", "Warning"]
    df["health_status"] = np.select(conditions, choices, default="Normal")
    return df


def get_active_alerts(df):
    """
    Return only Critical and Warning records as active alerts.
    Sorted by most severe first.
    """
    alerts = df[df["health_status"].isin(["Critical", "Warning"])].copy()
    alerts = alerts.sort_values(
        by="health_status",
        key=lambda x: x.map({"Critical": 0, "Warning": 1, "Normal": 2})
    )
    return alerts


# ============================================================
# LAYER 4: DASHBOARD — Streamlit UI Components
# ============================================================

def render_header():
    """Render the top header bar."""
    st.markdown("""
        <div class="pahms-header">
            <h2 style="color:#e2e8f0; margin:0;">
                ✈️ &nbsp; Predictive Aircraft Health Monitoring System
            </h2>
            <p style="color:#94a3b8; margin:4px 0 0 0; font-size:14px;">
                CIS-498 Capstone &nbsp;|&nbsp; Real-time Fleet Health Dashboard &nbsp;|&nbsp;
                Data Source: NASA ASRS Aviation Safety Reports
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_kpis(df):
    """
    Render the Fleet Summary KPI cards at the top of the dashboard.
    These give a quick snapshot of fleet health.
    """
    total = len(df)
    critical_count = len(df[df["health_status"] == "Critical"])
    warning_count  = len(df[df["health_status"] == "Warning"])
    normal_count   = len(df[df["health_status"] == "Normal"])
    alert_rate     = round((critical_count + warning_count) / total * 100, 1) if total > 0 else 0
    unique_models  = df["aircraft_model"].nunique()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("📋 Total Records", f"{total:,}")
    with col2:
        st.metric("🔴 Critical", f"{critical_count:,}", delta=None)
    with col3:
        st.metric("🟡 Warning", f"{warning_count:,}")
    with col4:
        st.metric("🟢 Normal", f"{normal_count:,}")
    with col5:
        st.metric("⚠️ Alert Rate", f"{alert_rate}%")
    with col6:
        st.metric("✈️ Aircraft Types", f"{unique_models}")


def render_health_charts(df):
    """
    Render charts showing health status distribution and trends.
    """
    col_left, col_right = st.columns(2)

    # ---- Donut Chart: Health Status Breakdown ----
    with col_left:
        st.subheader("🩺 Fleet Health Status Breakdown")
        status_counts = df["health_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]

        color_map = {
            "Critical": "#ef4444",
            "Warning":  "#f59e0b",
            "Normal":   "#22c55e"
        }

        fig_donut = px.pie(
            status_counts,
            names="Status",
            values="Count",
            hole=0.55,
            color="Status",
            color_discrete_map=color_map,
            template="plotly_dark"
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=20, b=10)
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_donut, use_container_width=True)

    # ---- Bar Chart: Incidents by Flight Phase ----
    with col_right:
        st.subheader("🛫 Incidents by Flight Phase")
        phase_data = (
            df.groupby(["flight_phase", "health_status"])
            .size()
            .reset_index(name="count")
        )
        top_phases = df["flight_phase"].value_counts().head(8).index
        phase_data = phase_data[phase_data["flight_phase"].isin(top_phases)]

        fig_bar = px.bar(
            phase_data,
            x="flight_phase",
            y="count",
            color="health_status",
            color_discrete_map=color_map,
            template="plotly_dark",
            barmode="stack",
            labels={"flight_phase": "Flight Phase", "count": "Record Count"}
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-30,
            legend_title="Health Status",
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)


def render_trend_charts(df):
    """
    Render time-series trend charts showing incident patterns over time.
    """
    col_left, col_right = st.columns(2)

    # ---- Line Chart: Monthly Trend ----
    with col_left:
        st.subheader("📈 Monthly Incident Trend")
        # Filter to years with enough data
        df_trend = df.dropna(subset=["report_date"])
        if df_trend.empty:
            st.info("No date data available for trend analysis.")
            return

        monthly = (
            df_trend.groupby([df_trend["report_date"].dt.to_period("Q"), "health_status"])
            .size()
            .reset_index(name="count")
        )
        monthly["report_date"] = monthly["report_date"].astype(str)

        color_map = {"Critical": "#ef4444", "Warning": "#f59e0b", "Normal": "#22c55e"}

        fig_trend = px.line(
            monthly,
            x="report_date",
            y="count",
            color="health_status",
            color_discrete_map=color_map,
            template="plotly_dark",
            markers=True,
            labels={"report_date": "Quarter", "count": "Incidents"}
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
            legend_title="Status",
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Bar Chart: Top Aircraft Models by Incident Count ----
    with col_right:
        st.subheader("✈️ Top Aircraft Models by Incidents")
        top_models = df["aircraft_model"].value_counts().head(10).reset_index()
        top_models.columns = ["Aircraft Model", "Incident Count"]

        fig_models = px.bar(
            top_models,
            x="Incident Count",
            y="Aircraft Model",
            orientation="h",
            template="plotly_dark",
            color="Incident Count",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            labels={"Aircraft Model": "", "Incident Count": "Number of Reports"}
        )
        fig_models.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(categoryorder="total ascending"),
            showlegend=False,
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig_models, use_container_width=True)


def render_component_chart(df):
    """
    Render a chart showing the most common component problems.
    """
    st.subheader("🔧 Component Problem Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        # Most common component statuses
        comp_status = df["component_status"].value_counts().head(8).reset_index()
        comp_status.columns = ["Component Status", "Count"]

        fig_comp = px.bar(
            comp_status,
            x="Count",
            y="Component Status",
            orientation="h",
            template="plotly_dark",
            color="Count",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
        )
        fig_comp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_right:
        # Root cause distribution
        root_cause = df["root_cause"].value_counts().head(8).reset_index()
        root_cause.columns = ["Root Cause", "Count"]

        fig_root = px.pie(
            root_cause,
            names="Root Cause",
            values="Count",
            template="plotly_dark",
            hole=0.4
        )
        fig_root.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_root, use_container_width=True)


def render_altitude_scatter(df):
    """
    Render a scatter plot of altitude vs flight hours, colored by health status.
    This simulates a sensor trend visualization.
    """
    st.subheader("📡 Altitude vs. Flight Hours — Sensor Trend View")

    # Filter first, then sample only as many rows as actually available
    filtered = df[df["cruise_altitude_ft"] > 0]

    if filtered.empty:
        st.info("No altitude data available for the current filter selection.")
        return

    sample = filtered.sample(min(3000, len(filtered)), random_state=42)

    color_map = {"Critical": "#ef4444", "Warning": "#f59e0b", "Normal": "#22c55e"}

    fig = px.scatter(
        sample,
        x="flight_hours",
        y="cruise_altitude_ft",
        color="health_status",
        color_discrete_map=color_map,
        template="plotly_dark",
        opacity=0.65,
        hover_data=["aircraft_model", "flight_phase", "component_status"],
        labels={
            "flight_hours": "Flight Hours",
            "cruise_altitude_ft": "Cruise Altitude (ft)",
            "health_status": "Status"
        }
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_alerts_panel(df):
    """
    Render the Active Alerts Panel.
    Shows Critical and Warning records that need maintenance attention.
    """
    st.subheader("🚨 Active Alerts Panel")

    alerts = get_active_alerts(df)

    if alerts.empty:
        st.success("✅ No active alerts in the current filter selection.")
        return

    # Summary row
    crit = len(alerts[alerts["health_status"] == "Critical"])
    warn = len(alerts[alerts["health_status"] == "Warning"])
    st.markdown(
        f"**{len(alerts):,} active alerts** — "
        f"🔴 {crit} Critical &nbsp;&nbsp; 🟡 {warn} Warning"
    )

    # Show top 50 alerts in a formatted table
    display_alerts = alerts[[
        "aircraft_id", "aircraft_model", "flight_phase",
        "health_status", "component_status", "anomaly_type",
        "base_location", "report_date"
    ]].head(50).copy()

    # Rename columns for display
    display_alerts.columns = [
        "Aircraft ID", "Model", "Flight Phase",
        "Status", "Component Issue", "Anomaly Type",
        "Location", "Report Date"
    ]
    display_alerts["Report Date"] = pd.to_datetime(
        display_alerts["Report Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    # Color rows by status
    def highlight_status(row):
        if row["Status"] == "Critical":
            return ["background-color: #3b0000; color: #fca5a5"] * len(row)
        elif row["Status"] == "Warning":
            return ["background-color: #2d1a00; color: #fde68a"] * len(row)
        return [""] * len(row)

    styled = display_alerts.style.apply(highlight_status, axis=1)
    st.dataframe(styled, use_container_width=True, height=300)


def render_health_table(df):
    """
    Render the full Aircraft Health Records table with filters applied.
    """
    st.subheader("📋 Aircraft Health Records")

    display = df[[
        "aircraft_id", "aircraft_model", "operator", "flight_phase",
        "health_status", "component_status", "root_cause",
        "base_location", "flight_hours", "report_date"
    ]].head(200).copy()

    display.columns = [
        "Aircraft ID", "Model", "Operator", "Flight Phase",
        "Health Status", "Component Status", "Root Cause",
        "Location", "Flight Hours", "Report Date"
    ]
    display["Report Date"] = pd.to_datetime(
        display["Report Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    st.dataframe(display, use_container_width=True, height=350)
    st.caption(f"Showing top 200 of {len(df):,} filtered records.")


def render_sidebar(df):
    """
    Render sidebar controls for filtering the dashboard.
    Returns filter values selected by the user.
    """
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/FAA_logo.svg/200px-FAA_logo.svg.png",
        width=80
    )
    st.sidebar.title("🔧 Filter Controls")
    st.sidebar.markdown("---")

    # ---- Aircraft Model Filter ----
    all_models = sorted(df["aircraft_model"].dropna().unique().tolist())
    selected_models = st.sidebar.multiselect(
        "✈️ Aircraft Model",
        options=all_models,
        default=[],
        placeholder="All models"
    )

    # ---- Flight Phase Filter ----
    all_phases = sorted(df["flight_phase"].dropna().unique().tolist())
    selected_phases = st.sidebar.multiselect(
        "🛫 Flight Phase",
        options=all_phases,
        default=[],
        placeholder="All phases"
    )

    # ---- Health Status Filter ----
    selected_statuses = st.sidebar.multiselect(
        "🩺 Health Status",
        options=["Critical", "Warning", "Normal"],
        default=["Critical", "Warning", "Normal"]
    )

    # ---- Year Range Filter ----
    min_year = int(df["report_year"].min()) if not df.empty else 2005
    max_year = int(df["report_year"].max()) if not df.empty else 2023
    year_range = st.sidebar.slider(
        "📅 Report Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "📌 **PAHMS Prototype**\n\n"
        "Data: NASA ASRS Aviation Safety Reports\n\n"
        "Built with Python + Streamlit\n\n"
        "CIS-498 Capstone Project"
    )

    return selected_models, selected_phases, selected_statuses, year_range


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """
    Main function — orchestrates all dashboard layers.
    Runs top to bottom when Streamlit refreshes.
    """

    # ---- Header ----
    render_header()

    # ---- Load Data (Layer 1: Data Input) ----
    with st.spinner("Loading aircraft safety records..."):
        df = load_data()

    # ---- Sidebar Filters (Layer 4: Dashboard Controls) ----
    selected_models, selected_phases, selected_statuses, year_range = render_sidebar(df)

    # ---- Apply Filters (Layer 2: Data Processing) ----
    filtered_df = filter_data(df, selected_models, selected_phases, selected_statuses, year_range)

    if filtered_df.empty:
        st.warning("⚠️ No records match the selected filters. Please adjust your filter settings.")
        return

    # ---- KPI Cards ----
    render_kpis(filtered_df)
    st.markdown("---")

    # ---- Tab Navigation ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Alerts",
        "📊 Health Overview",
        "📈 Trends & Sensors",
        "🔧 Components",
        "📋 Records Table"
    ])

    with tab1:
        render_alerts_panel(filtered_df)

    with tab2:
        render_health_charts(filtered_df)

    with tab3:
        render_trend_charts(filtered_df)
        st.markdown("---")
        render_altitude_scatter(filtered_df)

    with tab4:
        render_component_chart(filtered_df)

    with tab5:
        render_health_table(filtered_df)

    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#475569; font-size:12px;'>"
        "PAHMS — Predictive Aircraft Health Monitoring System &nbsp;|&nbsp; "
        "CIS-498 Capstone &nbsp;|&nbsp; "
        "Data: NASA Aviation Safety Reporting System (ASRS)"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
