"""Streamlit Monitoring Dashboard for Home Loan Approval ML System.

Displays operational inference metrics, real-time statistical drift score,
active champion model version, automated retraining history, and model
performance comparisons before and after retraining.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path when executed via streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from monitoring.metrics import DEFAULT_METRICS_PATH, metrics_tracker

st.set_page_config(
    page_title="Home Loan ML Monitoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_monitoring_records() -> Dict[str, Any]:
    """Fetch current monitoring records from persistent JSON store and memory."""
    # Ensure any recent data in file is synchronized
    metrics_tracker.load_from_json()
    return metrics_tracker.get_metrics()


def main():
    """Render the Home Loan ML Monitoring Streamlit Dashboard."""
    st.title("Home Loan ML Monitoring")
    st.caption("Continuous Observability, Drift Detection & Automated Retraining Governance")

    # Fetch up-to-date metrics
    records = load_monitoring_records()

    # -------------------------------------------------------------
    # 1. Primary Top Metrics Row (Requested Dashboard Specification)
    # -------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total predictions",
            value=f"{records['total_predictions']:,}",
            help="Total inference requests processed by the prediction service",
        )

    with col2:
        drift_score = records.get("last_drift_score", 0.0)
        st.metric(
            label="Drift score",
            value=f"{drift_score}%",
            delta=f"{drift_score}%" if drift_score > 0 else None,
            delta_color="inverse",
            help="Percentage of monitored input features exhibiting statistical drift",
        )

    with col3:
        st.metric(
            label="Model version",
            value=records.get("current_model_version", "champion-v1"),
            help="Active champion model version currently serving production inference",
        )

    with col4:
        st.metric(
            label="Retraining runs",
            value=f"{records.get('retraining_attempts', 0):,}",
            help="Total automated retraining executions triggered by drift",
        )

    st.divider()

    # -------------------------------------------------------------
    # 2. Statistical Data Drift Status
    # -------------------------------------------------------------
    st.subheader("Statistical Data Drift Status")

    drift_events_count = records.get("drift_events_count", 0)
    drift_status = records.get("last_drift_status", "NO DRIFT DETECTED")
    last_drift_time = records.get("last_drift_timestamp")

    if drift_events_count == 0:
        st.info(
            "ℹ️ **Status: Awaiting Initial Drift Evaluation** | No drift evaluation events recorded yet."
        )
    elif drift_status == "INCONCLUSIVE":
        st.warning(
            f"⚠️ **Status: {drift_status}** | No valid feature tests could be evaluated."
        )
    elif drift_status == "DRIFT DETECTED":
        st.error(
            f"⚠️ **Status: {drift_status}** | Drift Score: **{drift_score}%** of features drifted"
        )
    else:
        st.success(
            f"✅ **Status: {drift_status}** | Drift Score: **{drift_score}%** (Below 20% alert threshold)"
        )

    drift_meta_col1, drift_meta_col2 = st.columns(2)
    with drift_meta_col1:
        st.write(
            f"**Last Drift Evaluation:** {last_drift_time or 'Awaiting initial evaluation run'}"
        )
        st.write(f"**Total Drift Evaluation Events:** {drift_events_count}")

    with drift_meta_col2:
        drifted_features = records.get("last_drifted_features", [])
        if drifted_features:
            st.write(f"**Drifted Features Detected ({len(drifted_features)}):**")
            st.markdown(", ".join([f"`{feat}`" for feat in drifted_features]))
        else:
            st.write("**Drifted Features:** None detected")

    st.divider()

    # -------------------------------------------------------------
    # 3. Model Performance Before and After Retraining
    # -------------------------------------------------------------
    st.subheader("Automated Retraining & Model Performance")

    retrain_col1, retrain_col2, retrain_col3 = st.columns(3)
    champ_score = records.get("champion_performance")
    cand_score = records.get("candidate_performance")

    delta_val = (
        f"{cand_score - champ_score:+.4f}"
        if (cand_score is not None and champ_score is not None)
        else None
    )

    with retrain_col1:
        st.metric(
            label="Champion Performance (Before)",
            value=f"{champ_score:.4f}" if champ_score is not None else "N/A",
            help="Validation F1-score of the active champion model",
        )

    with retrain_col2:
        st.metric(
            label="Candidate Performance (After)",
            value=f"{cand_score:.4f}" if cand_score is not None else "N/A",
            delta=delta_val,
            help="Validation F1-score achieved by the retrained candidate model",
        )

    with retrain_col3:
        st.metric(
            label="Successful Promotions",
            value=f"{records.get('successful_promotions', 0)} / {records.get('retraining_attempts', 0)}",
            help="Count of retrained models that outperformed the champion and were deployed",
        )

    st.write(
        f"**Last Retraining Timestamp:** {records.get('last_retraining_time') or 'No retraining triggered yet'}"
    )

    # Retraining history log table
    history = records.get("retraining_history", [])
    if history:
        st.write("**Recent Retraining Runs:**")
        history_df = pd.DataFrame(history)
        cols_to_show = [
            c
            for c in [
                "timestamp",
                "retraining_run_id",
                "status",
                "success",
                "promoted",
                "champion_score",
                "candidate_score",
                "active_model_version",
            ]
            if c in history_df.columns
        ]
        st.dataframe(history_df[cols_to_show], use_container_width=True)
    else:
        st.info("No automated retraining executions recorded yet.")

    st.divider()

    # -------------------------------------------------------------
    # 4. Operational Serving Telemetry
    # -------------------------------------------------------------
    st.subheader("Operational Serving Telemetry")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.write(f"**Successful Predictions:** {records['successful_predictions']}")
        st.write(f"**Failed Requests:** {records['failed_predictions']}")
    with t2:
        st.write(f"**Approved Loans:** {records['approved_predictions']}")
        st.write(f"**Rejected Loans:** {records['rejected_predictions']}")
    with t3:
        st.write(f"**Approval Rate:** {records['approval_rate'] * 100:.1f}%")
        st.write(f"**Rejection Rate:** {records['rejection_rate'] * 100:.1f}%")
    with t4:
        st.write(f"**Average Latency:** {records['average_latency_ms']} ms")
        st.write(f"**Service Uptime:** {records['uptime_seconds']} s")

    # -------------------------------------------------------------
    # Sidebar
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("Dashboard Settings")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        st.caption(f"Metrics Store: `{DEFAULT_METRICS_PATH}`")
        st.caption(f"Server Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")


if __name__ == "__main__":
    main()
