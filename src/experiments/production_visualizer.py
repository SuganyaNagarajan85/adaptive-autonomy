"""
Production Visualizer — scenario-specific matplotlib plots.

Required outputs per scenario (saved to outputs/<scenario_name>/):
  autonomy.png        — B_t static vs adaptive over time
  incidents.png       — incident count/rate per step
  efficiency.png      — bar chart: efficiency comparison
  system_behavior.png — traffic + error_rate + autonomy on shared axes

Design principles:
  - matplotlib only (no seaborn, no external style sheets)
  - Consistent color language across all plots:
      adaptive  = #2E86AB (blue)
      static    = #E84855 (red)
      traffic   = #3BB273 (green)
      error     = #E84855 (red)
      anomaly   = #F4A261 (orange, shaded)
  - All figures tight_layout() — safe for CLI and Streamlit rendering
  - Saved at 150 DPI — readable without being huge
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.experiments.metrics import StepMetrics
from src.experiments.production_metrics import ScenarioMetrics, StepEfficiency
from src.experiments.production_runner import ScenarioExperimentResult
from src.simulation.production_simulator import ProductionSimulator

# ─── Color constants ─────────────────────────────────────────────────────────

_ADAPTIVE  = "#2E86AB"
_STATIC    = "#E84855"
_TRAFFIC   = "#3BB273"
_ERROR     = "#E84855"
_LATENCY   = "#9B5DE5"
_AUTONOMY  = "#2E86AB"
_ANOMALY   = "#F4A261"
_ANOMALY_A = 0.13
_GRID_A    = 0.25


class ProductionVisualizer:
    """
    Generates the four required scenario plots plus optional supplementary charts.

    Usage::

        result = runner.run_scenario(flash_sale_scenario())
        viz = ProductionVisualizer(output_dir="outputs")
        paths = viz.plot_all(result)
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        self._base_dir = Path(output_dir)

    # ─── Public API ──────────────────────────────────────────────────────

    def plot_all(self, result: ScenarioExperimentResult) -> List[str]:
        """
        Generate all four required plots for a scenario result.
        Returns list of saved file paths.
        """
        out_dir = self._scenario_dir(result.scenario.name)
        paths = [
            self.plot_autonomy(result, out_dir),
            self.plot_incidents(result, out_dir),
            self.plot_efficiency(result, out_dir),
            self.plot_system_behavior(result, out_dir),
        ]
        return [p for p in paths if p]

    # ─── Plot 1: Autonomy over time ───────────────────────────────────────

    def plot_autonomy(
        self,
        result: ScenarioExperimentResult,
        out_dir: Optional[Path] = None,
    ) -> str:
        """
        B_t for static (flat line) vs adaptive (evolving) over simulation steps.
        Anomaly periods are shaded.
        """
        out_dir = out_dir or self._scenario_dir(result.scenario.name)
        s_steps = result.static_steps
        a_steps = result.adaptive_steps

        fig, ax = plt.subplots(figsize=(13, 5))
        x = [s.step for s in a_steps]

        # Static: flat boundary
        s_b = [s.mean_boundary for s in s_steps]
        a_b = [s.mean_boundary for s in a_steps]

        ax.plot(x, s_b, color=_STATIC,   linewidth=2.0,
                linestyle="--", label=f"Static  B={s_b[0]:.3f} (fixed)", alpha=0.85)
        ax.plot(x, a_b, color=_ADAPTIVE, linewidth=2.2,
                label=f"Adaptive B: {a_b[0]:.3f} → {a_b[-1]:.3f}")

        self._shade_anomalies(ax, a_steps)
        self._annotate_boundary_events(ax, a_steps, a_b)

        ax.set_xlabel("Simulation Step", fontsize=11)
        ax.set_ylabel("Mean Autonomy Boundary B", fontsize=11)
        ax.set_title(
            f"Autonomy Boundary Over Time — {result.scenario.name.replace('_', ' ').title()}\n"
            f"(orange = anomaly period, adaptive system earned "
            f"{result.adaptive_metrics.mean_autonomy_utilization:.1%} auto utilisation)",
            fontsize=12,
        )
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=_GRID_A)
        fig.tight_layout()

        path = str(out_dir / "autonomy.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Plot 2: Incident trend ───────────────────────────────────────────

    def plot_incidents(
        self,
        result: ScenarioExperimentResult,
        out_dir: Optional[Path] = None,
    ) -> str:
        """
        Incident rate per step for static vs adaptive.
        Uses rolling mean to smooth noise. Anomaly periods shaded.
        """
        out_dir = out_dir or self._scenario_dir(result.scenario.name)
        s_steps = result.static_steps
        a_steps = result.adaptive_steps
        window  = max(1, len(a_steps) // 25)

        s_ir = self._rolling_mean([s.incident_rate for s in s_steps], window)
        a_ir = self._rolling_mean([s.incident_rate for s in a_steps], window)
        x    = list(range(len(a_ir)))

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

        # ── Top: incident rate ─────────────────────────────────────────
        axes[0].plot(x, s_ir, color=_STATIC,   linewidth=1.8,
                     label=f"Static  (mean={result.static_metrics.mean_incident_rate:.4f})")
        axes[0].plot(x, a_ir, color=_ADAPTIVE, linewidth=1.8,
                     label=f"Adaptive (mean={result.adaptive_metrics.mean_incident_rate:.4f})")
        self._shade_anomalies(axes[0], a_steps[:len(x)])
        axes[0].set_ylabel("Incident Rate (rolling avg)", fontsize=10)
        axes[0].set_title(
            f"Incident Trend — {result.scenario.name.replace('_', ' ').title()}  "
            f"[Reduction: {result.incident_reduction_pct:+.1f}%]",
            fontsize=12,
        )
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=_GRID_A)
        axes[0].set_ylim(bottom=0)

        # ── Bottom: cumulative incidents ───────────────────────────────
        s_cum = np.cumsum([s.incidents for s in s_steps])
        a_cum = np.cumsum([s.incidents for s in a_steps])
        x2    = list(range(len(s_cum)))
        axes[1].fill_between(x2, s_cum, a_cum,
                              where=(s_cum >= a_cum),
                              alpha=0.15, color=_STATIC, label="Static excess")
        axes[1].plot(x2, s_cum, color=_STATIC,   linewidth=1.8,
                     label=f"Static cumulative  ({result.static_metrics.total_incidents})")
        axes[1].plot(x2, a_cum, color=_ADAPTIVE, linewidth=1.8,
                     label=f"Adaptive cumulative ({result.adaptive_metrics.total_incidents})")
        self._shade_anomalies(axes[1], a_steps)
        axes[1].set_xlabel("Simulation Step", fontsize=10)
        axes[1].set_ylabel("Cumulative Incidents", fontsize=10)
        axes[1].set_title("Cumulative Incident Count", fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=_GRID_A)

        fig.tight_layout()
        path = str(out_dir / "incidents.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Plot 3: Efficiency comparison ───────────────────────────────────

    def plot_efficiency(
        self,
        result: ScenarioExperimentResult,
        out_dir: Optional[Path] = None,
    ) -> str:
        """
        Bar chart comparing efficiency metrics: static vs adaptive.
        efficiency = autonomy_utilization − incident_rate (higher is better).
        """
        out_dir = out_dir or self._scenario_dir(result.scenario.name)
        sm = result.static_metrics
        am = result.adaptive_metrics

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: bar chart of aggregate efficiency metrics ─────────────
        categories = [
            "Overall\nEfficiency",
            "Efficiency\n(Normal)",
            "Efficiency\n(Anomaly)",
        ]
        s_vals = [sm.mean_efficiency, sm.efficiency_during_normal, sm.efficiency_during_anomaly]
        a_vals = [am.mean_efficiency, am.efficiency_during_normal, am.efficiency_during_anomaly]

        x     = np.arange(len(categories))
        width = 0.35
        bars_s = axes[0].bar(x - width/2, s_vals, width, label="Static",
                              color=_STATIC,   alpha=0.85, edgecolor="white", linewidth=0.8)
        bars_a = axes[0].bar(x + width/2, a_vals, width, label="Adaptive",
                              color=_ADAPTIVE, alpha=0.85, edgecolor="white", linewidth=0.8)

        # Value labels on bars
        for bar in bars_s:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.005,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars_a:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.005,
                         f"{h:.3f}", ha="center", va="bottom", fontsize=8, color=_ADAPTIVE)

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, fontsize=10)
        axes[0].set_ylabel("Efficiency (Autonomy Util. − Incident Rate)", fontsize=10)
        axes[0].set_title(
            f"Efficiency Comparison\n"
            f"Gain: {result.efficiency_gain:+.4f}  |  "
            f"scenario: {result.scenario.name.replace('_', ' ').title()}",
            fontsize=11,
        )
        axes[0].legend(fontsize=10)
        axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        axes[0].grid(True, axis="y", alpha=_GRID_A)
        axes[0].set_ylim(
            min(min(s_vals), min(a_vals)) - 0.05,
            max(max(s_vals), max(a_vals)) + 0.08,
        )

        # ── Right: efficiency over time ─────────────────────────────────
        s_eff = [se.efficiency for se in result.static_metrics.step_efficiencies]
        a_eff = [se.efficiency for se in result.adaptive_metrics.step_efficiencies]
        win   = max(1, len(a_eff) // 25)
        s_eff_s = self._rolling_mean(s_eff, win)
        a_eff_s = self._rolling_mean(a_eff, win)
        x2    = list(range(len(s_eff_s)))

        axes[1].plot(x2, s_eff_s, color=_STATIC,   linewidth=1.8, label="Static")
        axes[1].plot(x2, a_eff_s, color=_ADAPTIVE, linewidth=1.8, label="Adaptive")
        axes[1].fill_between(x2, s_eff_s, a_eff_s,
                              where=[a > s for a, s in zip(a_eff_s, s_eff_s)],
                              alpha=0.12, color=_ADAPTIVE, label="Adaptive advantage")
        self._shade_anomalies(axes[1], result.adaptive_steps)
        axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Simulation Step", fontsize=10)
        axes[1].set_ylabel("Efficiency (rolling avg)", fontsize=10)
        axes[1].set_title("Efficiency Over Time", fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=_GRID_A)

        fig.tight_layout()
        path = str(out_dir / "efficiency.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Plot 4: System behavior ──────────────────────────────────────────

    def plot_system_behavior(
        self,
        result: ScenarioExperimentResult,
        out_dir: Optional[Path] = None,
    ) -> str:
        """
        Three-panel graph on shared x-axis:
          Panel 1: traffic (green)
          Panel 2: error_rate (red)
          Panel 3: autonomy boundary B — static (dashed red) vs adaptive (blue)

        Also overlays anomaly period shading on all three panels.
        Uses the ProductionSimulator replay log from the adaptive run's prod_sim.
        """
        out_dir = out_dir or self._scenario_dir(result.scenario.name)

        # Regenerate environment series from scenario (both runs share same pattern)
        prod_sim = ProductionSimulator(result.scenario)
        env_series = [prod_sim.step(t) for t in range(result.scenario.total_steps)]

        x       = list(range(len(env_series)))
        traffic = [e.traffic for e in env_series]
        errors  = [e.error_rate for e in env_series]
        latency = [e.latency_ms for e in env_series]

        s_b = [s.mean_boundary for s in result.static_steps]
        a_b = [s.mean_boundary for s in result.adaptive_steps]

        fig = plt.figure(figsize=(14, 9))
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08,
                                top=0.93, bottom=0.07, left=0.08, right=0.97)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)

        # ── Panel 1: Traffic ────────────────────────────────────────────
        ax1.plot(x, traffic, color=_TRAFFIC, linewidth=1.6, alpha=0.9)
        ax1.fill_between(x, 0, traffic, color=_TRAFFIC, alpha=0.08)
        ax1.set_ylabel("Traffic\n(requests/step)", fontsize=9)
        ax1.set_title(
            f"System Behavior — {result.scenario.name.replace('_', ' ').title()}",
            fontsize=13,
        )
        ax1.grid(True, alpha=_GRID_A)
        self._shade_anomalies(ax1, result.adaptive_steps)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # ── Panel 2: Error rate ─────────────────────────────────────────
        high_thr = result.scenario.high_error_threshold
        mod_thr  = result.scenario.moderate_error_threshold

        ax2.plot(x, errors, color=_ERROR, linewidth=1.6, alpha=0.9, label="Error rate")
        ax2.fill_between(x, 0, errors, color=_ERROR, alpha=0.07)
        ax2.axhline(high_thr, color="darkred", linestyle="--", linewidth=1.2,
                    alpha=0.7, label=f"High error τ={high_thr:.2f}")
        ax2.axhline(mod_thr,  color="orange",  linestyle=":",  linewidth=1.0,
                    alpha=0.7, label=f"Moderate error τ={mod_thr:.2f}")
        ax2.set_ylabel("Error Rate", fontsize=9)
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_ylim(-0.01, min(1.05, max(errors) * 1.3 + 0.05))
        ax2.grid(True, alpha=_GRID_A)
        self._shade_anomalies(ax2, result.adaptive_steps)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # ── Panel 3: Autonomy boundary ──────────────────────────────────
        ax3.plot(x, s_b, color=_STATIC,   linestyle="--", linewidth=1.8,
                 label=f"Static B (fixed={s_b[0]:.3f})", alpha=0.8)
        ax3.plot(x, a_b, color=_ADAPTIVE, linewidth=2.0,
                 label=f"Adaptive B ({a_b[0]:.3f} → {a_b[-1]:.3f})")
        ax3.set_ylabel("Autonomy\nBoundary B", fontsize=9)
        ax3.set_xlabel("Simulation Step", fontsize=10)
        ax3.set_ylim(-0.02, 1.05)
        ax3.legend(loc="upper left", fontsize=9)
        ax3.grid(True, alpha=_GRID_A)
        self._shade_anomalies(ax3, result.adaptive_steps)

        # Shared anomaly legend patch
        anomaly_patch = mpatches.Patch(color=_ANOMALY, alpha=0.5, label="Anomaly period")
        ax3.legend(
            handles=ax3.get_legend_handles_labels()[0] + [anomaly_patch],
            labels=ax3.get_legend_handles_labels()[1] + ["Anomaly period"],
            loc="upper left", fontsize=9,
        )

        # GridSpec manual margins handle layout — skip tight_layout to avoid warning
        path = str(out_dir / "system_behavior.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─── Private helpers ─────────────────────────────────────────────────

    def _scenario_dir(self, scenario_name: str) -> Path:
        d = self._base_dir / scenario_name.lower().replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _rolling_mean(series: List[float], window: int) -> List[float]:
        result = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            result.append(sum(series[start:i + 1]) / (i - start + 1))
        return result

    @staticmethod
    def _shade_anomalies(ax, steps: List[StepMetrics]) -> None:
        in_anomaly = False
        start = 0
        for s in steps:
            if s.is_anomaly and not in_anomaly:
                start = s.step
                in_anomaly = True
            elif not s.is_anomaly and in_anomaly:
                ax.axvspan(start, s.step, alpha=_ANOMALY_A, color=_ANOMALY)
                in_anomaly = False
        if in_anomaly and steps:
            ax.axvspan(start, steps[-1].step, alpha=_ANOMALY_A, color=_ANOMALY)

    @staticmethod
    def _annotate_boundary_events(
        ax, steps: List[StepMetrics], b_series: List[float]
    ) -> None:
        """Mark the largest single-step boundary drop (contraction event)."""
        if len(b_series) < 2:
            return
        max_drop = 0.0
        max_drop_step = 0
        for i in range(1, len(b_series)):
            delta = b_series[i - 1] - b_series[i]
            if delta > max_drop:
                max_drop = delta
                max_drop_step = steps[i].step
        if max_drop > 0.01:
            ax.annotate(
                f"⬇ max contraction\nΔB=−{max_drop:.3f}",
                xy=(max_drop_step, b_series[max_drop_step]),
                xytext=(max_drop_step + len(b_series) // 15, b_series[max_drop_step] + 0.1),
                fontsize=8,
                color=_STATIC,
                arrowprops=dict(arrowstyle="->", color=_STATIC, lw=1.2),
            )
