"""
Streamlit Dashboard — Human-in-the-Loop UI for the Autonomy Learning Loop.

Features:
  - Real-time autonomy boundary visualization per decision class
  - Human review queue with approve/override actions
  - Risk score S_t monitoring panel
  - Audit event feed
  - Shadow mode comparison panel
  - Run experiments and view results inline

Run:  streamlit run ui/dashboard.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.autonomy_controller.controller import AutonomyController
from src.decision_engine.models import DecisionType
from src.experiments.runner import ExperimentRunner
from src.experiments.visualizer import Visualizer
from src.governance.audit_logger import AuditLogger
from src.governance.human_review import HumanReviewQueue, ReviewStatus
from src.simulation.simulator import Simulator

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomy Learning Loop",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state ───────────────────────────────────────────────────────────
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "experiment_result" not in st.session_state:
    st.session_state.experiment_result = None
if "controller" not in st.session_state:
    st.session_state.controller = AutonomyController(
        initial_boundary=0.5, shadow_mode=False
    )
if "review_queue" not in st.session_state:
    st.session_state.review_queue = HumanReviewQueue(capacity=200, sla_seconds=300)
if "audit_logger" not in st.session_state:
    st.session_state.audit_logger = AuditLogger(
        log_path="logs/dashboard_audit.jsonl", enabled=True
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Autonomy Loop")
    st.caption("Feedback-Driven Autonomy System")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔬 Run Experiment", "📋 Review Queue", "📊 Boundary Monitor", "🔍 Audit Log"],
    )


# ─── Overview ────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("Feedback-Driven Autonomy: Learning Safe Automation Boundaries")
    st.markdown("""
    This system implements the **Autonomy Learning Loop (ALL)** from *Suganya Nagarajan (2024)*.

    ### Core Concept
    Autonomy boundary **B ∈ [0,1]** evolves based on operational feedback signals:

    | Condition | Update Rule | Effect |
    |-----------|-------------|--------|
    | S_t < τ_safe | ΔB = α(1−B) | Slow expansion |
    | S_t ≥ τ_risk | ΔB = −βB | Fast contraction |
    | τ_safe ≤ S_t < τ_risk | ΔB = 0 | Hold |

    **β >> α** — asymmetric safety guarantee: contracts fast, expands slowly.
    """)

    st.subheader("Current Autonomy Boundaries")
    controller = st.session_state.controller
    cols = st.columns(3)
    for i, dt in enumerate(DecisionType):
        b = controller.get_boundary(dt)
        with cols[i % 3]:
            st.metric(
                label=dt.value.replace("_", " ").title(),
                value=f"{b:.3f}",
                delta=f"{'Frozen' if controller.is_frozen(dt) else 'Active'}",
            )
            st.progress(b)

    st.divider()
    st.subheader("About the System")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Feedback Signals**\nRollback · Override · Incident · Trust Degradation · Confidence Divergence")
    with col2:
        st.warning("**Safety Principles**\nAsymmetric updates · Rate-limited · Window-based · Attribution-tolerant")
    with col3:
        st.success("**E-commerce Context**\nRecommendations · Offers · Fraud · Pricing · Notifications · Search")


# ─── Run Experiment ──────────────────────────────────────────────────────────
elif page == "🔬 Run Experiment":
    st.title("🔬 Run Experiment: Static vs Adaptive Comparison")

    with st.expander("⚙️ Simulation Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            total_steps = st.slider("Total Steps", 100, 1000, 500, 50)
            decisions_per_step = st.slider("Decisions/Step", 5, 100, 20, 5)
            initial_boundary = st.slider("Initial Boundary B₀", 0.1, 0.9, 0.5, 0.05)
        with col2:
            alpha = st.slider("α (expansion rate)", 0.01, 0.20, 0.05, 0.01)
            beta = st.slider("β (contraction rate)", 0.10, 0.60, 0.30, 0.05)
            base_failure_rate = st.slider("Base Failure Rate", 0.01, 0.30, 0.08, 0.01)
        with col3:
            safe_threshold = st.slider("τ_safe", 0.10, 0.40, 0.25, 0.05)
            risk_threshold = st.slider("τ_risk", 0.40, 0.90, 0.60, 0.05)
            anomaly_probability = st.slider("Anomaly Probability", 0.01, 0.20, 0.05, 0.01)
            seed = st.number_input("Random Seed", value=42, min_value=0)

    if alpha >= beta:
        st.error("⚠️ β must be greater than α for the asymmetric safety guarantee (β >> α)")
    elif safe_threshold >= risk_threshold:
        st.error("⚠️ τ_safe must be less than τ_risk")
    elif st.button("▶️ Run Experiment", type="primary"):
        with st.spinner("Running static and adaptive simulations..."):
            runner = ExperimentRunner()
            result = runner.run_comparison(
                total_steps=total_steps,
                decisions_per_step=decisions_per_step,
                initial_boundary=initial_boundary,
                alpha=alpha,
                beta=beta,
                base_failure_rate=base_failure_rate,
                safe_threshold=safe_threshold,
                risk_threshold=risk_threshold,
                anomaly_probability=anomaly_probability,
                seed=int(seed),
            )
            st.session_state.experiment_result = result
            visualizer = Visualizer(output_dir="outputs")
            plot_paths = visualizer.plot_all(result)
            st.session_state.plot_paths = plot_paths
        st.success("Experiment complete!")

    result = st.session_state.experiment_result
    if result:
        st.divider()
        st.subheader("📊 Results Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Incident Reduction",
                f"{result.incident_reduction_pct:+.1f}%",
                delta_color="normal",
            )
        with col2:
            st.metric(
                "Override Reduction",
                f"{result.override_reduction_pct:+.1f}%",
                delta_color="normal",
            )
        with col3:
            st.metric(
                "Adaptive Mean Incident Rate",
                f"{result.adaptive_summary.mean_incident_rate:.4f}",
                delta=f"{result.adaptive_summary.mean_incident_rate - result.static_summary.mean_incident_rate:+.4f}",
            )
        with col4:
            st.metric(
                "Anomaly Resilience",
                f"{result.adaptive_summary.anomaly_resilience_score:.3f}",
                delta=f"{result.adaptive_summary.anomaly_resilience_score - result.static_summary.anomaly_resilience_score:+.3f}",
            )

        # Comparison table
        st.subheader("Detailed Comparison")
        comparison_data = {
            "Metric": [
                "Mean Incident Rate", "Mean Override Rate", "Mean Rollback Rate",
                "Autonomy Utilization", "Final Mean Boundary", "Boundary Stability σ",
                "Total Incidents", "Anomaly Resilience",
            ],
            "Static": [
                result.static_summary.mean_incident_rate,
                result.static_summary.mean_override_rate,
                result.static_summary.mean_rollback_rate,
                result.static_summary.mean_autonomy_utilization,
                result.static_summary.final_mean_boundary,
                result.static_summary.boundary_stability,
                result.static_summary.total_incidents,
                result.static_summary.anomaly_resilience_score,
            ],
            "Adaptive": [
                result.adaptive_summary.mean_incident_rate,
                result.adaptive_summary.mean_override_rate,
                result.adaptive_summary.mean_rollback_rate,
                result.adaptive_summary.mean_autonomy_utilization,
                result.adaptive_summary.final_mean_boundary,
                result.adaptive_summary.boundary_stability,
                result.adaptive_summary.total_incidents,
                result.adaptive_summary.anomaly_resilience_score,
            ],
        }
        st.dataframe(pd.DataFrame(comparison_data).round(4), use_container_width=True)

        # Show plots
        st.divider()
        st.subheader("📈 Visualizations")
        plot_paths = getattr(st.session_state, "plot_paths", [])
        for path in plot_paths:
            if Path(path).exists():
                st.image(path, use_container_width=True)


# ─── Review Queue ─────────────────────────────────────────────────────────────
elif page == "📋 Review Queue":
    st.title("📋 Human-in-the-Loop Review Queue")
    st.caption("Review decisions that exceeded the autonomy boundary and require human judgment.")

    queue: HumanReviewQueue = st.session_state.review_queue
    stats = queue.stats

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Queue Depth", stats["queue_depth"])
    with col2:
        st.metric("Total Enqueued", stats["total_enqueued"])
    with col3:
        st.metric("Total Resolved", stats["total_resolved"])
    with col4:
        st.metric("Override Rate", f"{stats['override_rate']:.2%}")

    st.divider()
    pending = queue.pending_items()
    if not pending:
        st.info("No pending review items. Run a simulation to populate the queue.")
    else:
        st.subheader(f"Pending Items ({len(pending)})")
        for item in pending[:20]:
            with st.expander(f"[{item['decision_type']}] {item['item_id'][:8]}... — Priority {abs(item['priority'])}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {item['decision_type']}")
                    st.write(f"**Age:** {item['age_seconds']:.0f}s")
                    st.write(f"**SLA Deadline:** {item['sla_deadline']}")
                    if item["is_expired"]:
                        st.error("SLA EXPIRED")
                with col2:
                    action = st.radio(
                        "Action",
                        ["APPROVE", "OVERRIDE"],
                        key=f"action_{item['item_id']}",
                    )
                    note = st.text_input("Note", key=f"note_{item['item_id']}")
                    if st.button("Submit", key=f"submit_{item['item_id']}"):
                        status = ReviewStatus.APPROVED if action == "APPROVE" else ReviewStatus.OVERRIDDEN
                        queue.resolve(
                            item["item_id"],
                            status=status,
                            reviewer="dashboard_operator",
                            note=note,
                        )
                        st.success(f"Resolved as {action}")
                        st.rerun()

    if st.button("🗑️ Expire Stale Items"):
        expired = queue.expire_stale()
        st.info(f"Expired {expired} stale items")
        st.rerun()


# ─── Boundary Monitor ────────────────────────────────────────────────────────
elif page == "📊 Boundary Monitor":
    st.title("📊 Live Boundary Monitor")

    result = st.session_state.experiment_result
    if result is None:
        st.warning("Run an experiment first to see boundary evolution.")
    else:
        decision_type = st.selectbox(
            "Select Decision Class",
            [dt.value for dt in DecisionType],
        )

        # Boundary over time for selected class
        steps = result.adaptive_steps
        x = [s.step for s in steps]
        b_series = [s.boundary_snapshots.get(decision_type, 0.0) for s in steps]
        s_t_series = [s.window_metrics.get(decision_type, {}).get("S_t", 0.0) for s in steps]
        anomaly_flags = [s.is_anomaly for s in steps]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        # Boundary
        axes[0].plot(x, b_series, color="#2E86AB", linewidth=2, label=f"B — {decision_type}")
        for i, (a, xi) in enumerate(zip(anomaly_flags, x)):
            if a:
                axes[0].axvline(xi, color="#F4A261", alpha=0.1, linewidth=0.5)
        axes[0].set_ylabel("Autonomy Boundary B")
        axes[0].set_title(f"Boundary Evolution: {decision_type}")
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Risk score
        axes[1].plot(x, s_t_series, color="#E84855", linewidth=1.5, alpha=0.8, label="S_t")
        axes[1].axhline(0.25, color="green", linestyle="--", alpha=0.6, linewidth=1.2, label="τ_safe")
        axes[1].axhline(0.60, color="red", linestyle="--", alpha=0.6, linewidth=1.2, label="τ_risk")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Risk Score S_t")
        axes[1].set_title("Risk Score S_t (triggers boundary updates)")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Manual boundary override
        st.divider()
        st.subheader("Manual Boundary Override")
        controller = st.session_state.controller
        dt_enum = DecisionType(decision_type)
        current_b = controller.get_boundary(dt_enum)
        new_b = st.slider(
            f"Set boundary for {decision_type}",
            0.0, 1.0, float(current_b), 0.01,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Apply Override"):
                controller.set_boundary(dt_enum, new_b)
                st.success(f"Boundary set to {new_b:.3f}")
        with col2:
            if st.button("Freeze Expansion"):
                controller.freeze(dt_enum)
                st.warning(f"{decision_type} expansion frozen")
        with col3:
            if st.button("Unfreeze"):
                controller.unfreeze(dt_enum)
                st.info(f"{decision_type} unfrozen")


# ─── Audit Log ───────────────────────────────────────────────────────────────
elif page == "🔍 Audit Log":
    st.title("🔍 Audit Log")
    st.caption("Complete audit trail of boundary updates, overrides, and governance events.")

    audit_logger: AuditLogger = st.session_state.audit_logger
    recent = audit_logger.recent_events(100)

    if not recent:
        st.info("No audit events yet. Run a simulation or use the API.")
    else:
        st.metric("Total Events", audit_logger.event_count())
        df = pd.DataFrame(recent)
        st.dataframe(df, use_container_width=True)

    if st.button("🔄 Refresh"):
        st.rerun()
