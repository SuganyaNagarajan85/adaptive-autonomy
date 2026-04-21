"""
Visualizer — generates all experiment plots.

Plots produced:
  1. Autonomy boundary B over time (per decision class, adaptive)
  2. Incidents vs autonomy boundary (adaptive)
  3. Static vs adaptive incident rate comparison
  4. Risk score S_t over time
  5. Boundary stability comparison (box plots)
  6. Anomaly period overlay on boundary timeline
  7. Override/rollback rates comparison
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for CLI and server use
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.decision_engine.models import DecisionType
from src.experiments.metrics import StepMetrics
from src.experiments.runner import ExperimentResult

# Color palette
_ADAPTIVE_COLOR = "#2E86AB"
_STATIC_COLOR   = "#E84855"
_ANOMALY_ALPHA  = 0.15
_ANOMALY_COLOR  = "#F4A261"

_DT_COLORS = {
    DecisionType.PRODUCT_RECOMMENDATION: "#2E86AB",
    DecisionType.SEARCH_RANKING:         "#3BB273",
    DecisionType.NOTIFICATION:           "#F4A261",
    DecisionType.PRICING:                "#9B5DE5",
    DecisionType.OFFER_SELECTION:        "#F15BB5",
    DecisionType.FRAUD_DETECTION:        "#E84855",
}


class Visualizer:

    def __init__(self, output_dir: str = "outputs") -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    def plot_all(self, result: ExperimentResult) -> List[str]:
        """Generate all plots and return list of saved file paths."""
        paths = []
        paths.append(self.plot_boundary_over_time(result))
        paths.append(self.plot_incidents_vs_boundary(result))
        paths.append(self.plot_static_vs_adaptive(result))
        paths.append(self.plot_risk_score(result))
        paths.append(self.plot_stability(result))
        paths.append(self.plot_override_rollback(result))
        return [p for p in paths if p]

    # ------------------------------------------------------------------

    def plot_boundary_over_time(self, result: ExperimentResult) -> str:
        """Plot autonomy boundary B for each decision class over simulation steps."""
        steps = result.adaptive_steps
        if not steps:
            return ""

        fig, ax = plt.subplots(figsize=(14, 6))

        x = [s.step for s in steps]
        for dt in DecisionType:
            series = [s.boundary_snapshots.get(dt.value, 0.0) for s in steps]
            ax.plot(x, series, label=dt.value.replace("_", " ").title(),
                    color=_DT_COLORS[dt], linewidth=1.8, alpha=0.85)

        # Shade anomaly periods
        self._shade_anomalies(ax, steps)

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Autonomy Boundary B")
        ax.set_title("Adaptive Autonomy Boundary B per Decision Class Over Time\n"
                     "(shaded = anomaly/flash-sale periods)", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="Initial B=0.5")
        fig.tight_layout()

        path = str(self._out / "01_boundary_over_time.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_incidents_vs_boundary(self, result: ExperimentResult) -> str:
        """Scatter plot: incident rate vs mean boundary (adaptive system)."""
        steps = result.adaptive_steps
        if not steps:
            return ""

        fig, ax = plt.subplots(figsize=(8, 6))

        x = [s.mean_boundary for s in steps]
        y = [s.incident_rate for s in steps]
        colors = [_ANOMALY_COLOR if s.is_anomaly else _ADAPTIVE_COLOR for s in steps]

        ax.scatter(x, y, c=colors, alpha=0.6, s=20, edgecolors="none")

        # Trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(x), max(x), 100)
            ax.plot(xs, p(xs), "k--", alpha=0.5, linewidth=1.5, label="Trend")

        ax.set_xlabel("Mean Autonomy Boundary B")
        ax.set_ylabel("Incident Rate")
        ax.set_title("Incident Rate vs Autonomy Boundary (Adaptive System)", fontsize=13)
        normal_patch = mpatches.Patch(color=_ADAPTIVE_COLOR, label="Normal Period")
        anomaly_patch = mpatches.Patch(color=_ANOMALY_COLOR, label="Anomaly Period")
        ax.legend(handles=[normal_patch, anomaly_patch])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = str(self._out / "02_incidents_vs_boundary.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_static_vs_adaptive(self, result: ExperimentResult) -> str:
        """Side-by-side comparison of incident rate: static vs adaptive."""
        static_steps = result.static_steps
        adaptive_steps = result.adaptive_steps
        if not static_steps or not adaptive_steps:
            return ""

        # Smooth with rolling mean for readability
        window = max(1, len(static_steps) // 20)

        s_ir = self._rolling_mean([s.incident_rate for s in static_steps], window)
        a_ir = self._rolling_mean([s.incident_rate for s in adaptive_steps], window)
        x = list(range(len(s_ir)))

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Incident rate comparison
        axes[0].plot(x, s_ir, color=_STATIC_COLOR, linewidth=1.8,
                     label=f"Static (mean={result.static_summary.mean_incident_rate:.4f})")
        axes[0].plot(x, a_ir, color=_ADAPTIVE_COLOR, linewidth=1.8,
                     label=f"Adaptive (mean={result.adaptive_summary.mean_incident_rate:.4f})")
        self._shade_anomalies(axes[0], adaptive_steps[:len(x)])
        axes[0].set_ylabel("Incident Rate (rolling avg)")
        axes[0].set_title(
            f"Incident Rate: Static vs Adaptive  |  "
            f"Reduction: {result.incident_reduction_pct:+.1f}%", fontsize=13
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Boundary comparison
        a_b = [s.mean_boundary for s in adaptive_steps]
        s_b = [s.mean_boundary for s in static_steps]
        x2 = list(range(len(adaptive_steps)))
        axes[1].plot(x2, a_b, color=_ADAPTIVE_COLOR, linewidth=1.8, label="Adaptive B")
        axes[1].plot(x2, s_b, color=_STATIC_COLOR, linewidth=1.8,
                     linestyle="--", label="Static B (fixed)")
        self._shade_anomalies(axes[1], adaptive_steps)
        axes[1].set_xlabel("Simulation Step")
        axes[1].set_ylabel("Mean Autonomy Boundary B")
        axes[1].set_title("Boundary Evolution: Adaptive Contracts During Anomalies", fontsize=13)
        axes[1].legend()
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        path = str(self._out / "03_static_vs_adaptive.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_risk_score(self, result: ExperimentResult) -> str:
        """Plot risk score S_t over time for each decision class."""
        steps = result.adaptive_steps
        if not steps:
            return ""

        fig, ax = plt.subplots(figsize=(14, 5))
        x = [s.step for s in steps]

        for dt in DecisionType:
            series = []
            for s in steps:
                wm = s.window_metrics.get(dt.value, {})
                series.append(wm.get("S_t", 0.0))
            ax.plot(x, series, label=dt.value.replace("_", " ").title(),
                    color=_DT_COLORS[dt], linewidth=1.5, alpha=0.75)

        # Threshold lines
        ax.axhline(0.25, color="green", linestyle="--", alpha=0.6, linewidth=1.2, label="τ_safe=0.25")
        ax.axhline(0.60, color="red",   linestyle="--", alpha=0.6, linewidth=1.2, label="τ_risk=0.60")
        self._shade_anomalies(ax, steps)

        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Risk Score S_t")
        ax.set_title("Aggregated Risk Score S_t per Decision Class\n"
                     "(green dashed = safe threshold, red dashed = risk threshold)", fontsize=13)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = str(self._out / "04_risk_score_over_time.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_stability(self, result: ExperimentResult) -> str:
        """Box plot comparing boundary distributions: static vs adaptive."""
        static_boundaries = [s.mean_boundary for s in result.static_steps]
        adaptive_boundaries = [s.mean_boundary for s in result.adaptive_steps]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Box plots
        axes[0].boxplot(
            [static_boundaries, adaptive_boundaries],
            labels=["Static", "Adaptive"],
            patch_artist=True,
            boxprops=dict(facecolor="#f0f0f0"),
        )
        axes[0].set_ylabel("Mean Boundary B")
        axes[0].set_title("Boundary Distribution: Stability Comparison")
        axes[0].grid(True, alpha=0.3)

        # Histogram overlay
        axes[1].hist(static_boundaries, bins=30, alpha=0.6, color=_STATIC_COLOR, label="Static")
        axes[1].hist(adaptive_boundaries, bins=30, alpha=0.6, color=_ADAPTIVE_COLOR, label="Adaptive")
        axes[1].set_xlabel("Mean Boundary B")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(
            f"Boundary Distribution Histogram\n"
            f"Static σ={result.static_summary.boundary_stability:.4f}  "
            f"Adaptive σ={result.adaptive_summary.boundary_stability:.4f}"
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        path = str(self._out / "05_boundary_stability.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_override_rollback(self, result: ExperimentResult) -> str:
        """Compare override and rollback rates across static vs adaptive."""
        window = max(1, len(result.static_steps) // 20)

        s_or = self._rolling_mean([s.override_rate for s in result.static_steps], window)
        a_or = self._rolling_mean([s.override_rate for s in result.adaptive_steps], window)
        s_rb = self._rolling_mean([s.rollback_rate for s in result.static_steps], window)
        a_rb = self._rolling_mean([s.rollback_rate for s in result.adaptive_steps], window)

        x = list(range(len(s_or)))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(x, s_or, color=_STATIC_COLOR, label="Static", linewidth=1.8)
        axes[0].plot(x, a_or, color=_ADAPTIVE_COLOR, label="Adaptive", linewidth=1.8)
        axes[0].set_title("Override Rate: Static vs Adaptive")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Override Rate")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, s_rb, color=_STATIC_COLOR, label="Static", linewidth=1.8)
        axes[1].plot(x, a_rb, color=_ADAPTIVE_COLOR, label="Adaptive", linewidth=1.8)
        axes[1].set_title("Rollback Rate: Static vs Adaptive")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Rollback Rate")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"Override Reduction: {result.override_reduction_pct:+.1f}%", fontsize=13
        )
        fig.tight_layout()
        path = str(self._out / "06_override_rollback_rates.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(series: list, window: int) -> list:
        result = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            result.append(sum(series[start:i + 1]) / (i - start + 1))
        return result

    @staticmethod
    def _shade_anomalies(ax, steps: List[StepMetrics]) -> None:
        """Shade anomaly periods on an axes."""
        in_anomaly = False
        start = 0
        for s in steps:
            if s.is_anomaly and not in_anomaly:
                start = s.step
                in_anomaly = True
            elif not s.is_anomaly and in_anomaly:
                ax.axvspan(start, s.step, alpha=_ANOMALY_ALPHA,
                           color=_ANOMALY_COLOR, label="_nolegend_")
                in_anomaly = False
        if in_anomaly and steps:
            ax.axvspan(start, steps[-1].step, alpha=_ANOMALY_ALPHA,
                       color=_ANOMALY_COLOR)
