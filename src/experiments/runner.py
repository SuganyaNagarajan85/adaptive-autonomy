"""
Experiment Runner — orchestrates the static vs adaptive comparison experiments.

Runs two parallel simulation configurations:
  1. Static system: B fixed at initial_boundary throughout (no feedback loop)
  2. Adaptive system: B evolves via the Autonomy Learning Loop

Produces ExperimentResult with metrics from both runs for comparison.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.experiments.metrics import ExperimentSummary, MetricsCollector, StepMetrics
from src.simulation.simulator import Simulator

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a comparison experiment (static vs adaptive)."""
    static_summary: ExperimentSummary
    adaptive_summary: ExperimentSummary
    static_steps: List[StepMetrics]
    adaptive_steps: List[StepMetrics]
    config: Dict[str, Any] = field(default_factory=dict)
    run_duration_seconds: float = 0.0

    @property
    def incident_reduction_pct(self) -> float:
        """Percentage reduction in incident rate: adaptive vs static."""
        static_ir = self.static_summary.mean_incident_rate
        adaptive_ir = self.adaptive_summary.mean_incident_rate
        if static_ir == 0:
            return 0.0
        return round(((static_ir - adaptive_ir) / static_ir) * 100, 2)

    @property
    def override_reduction_pct(self) -> float:
        static_or = self.static_summary.mean_override_rate
        adaptive_or = self.adaptive_summary.mean_override_rate
        if static_or == 0:
            return 0.0
        return round(((static_or - adaptive_or) / static_or) * 100, 2)

    def print_comparison(self) -> None:
        """Pretty-print the comparison table."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Static vs Adaptive Autonomy — Experiment Results", show_header=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Static System", justify="right")
        table.add_column("Adaptive System", justify="right")
        table.add_column("Improvement", justify="right", style="green")

        s = self.static_summary
        a = self.adaptive_summary

        rows = [
            ("Mean Incident Rate",      f"{s.mean_incident_rate:.4f}",      f"{a.mean_incident_rate:.4f}",      f"{self.incident_reduction_pct:+.1f}%"),
            ("Mean Override Rate",      f"{s.mean_override_rate:.4f}",      f"{a.mean_override_rate:.4f}",      f"{self.override_reduction_pct:+.1f}%"),
            ("Mean Rollback Rate",      f"{s.mean_rollback_rate:.4f}",      f"{a.mean_rollback_rate:.4f}",      "—"),
            ("Autonomy Utilization",    f"{s.mean_autonomy_utilization:.4f}", f"{a.mean_autonomy_utilization:.4f}", "—"),
            ("Final Mean Boundary",     f"{s.final_mean_boundary:.4f}",     f"{a.final_mean_boundary:.4f}",     "—"),
            ("Boundary Stability σ",    f"{s.boundary_stability:.4f}",      f"{a.boundary_stability:.4f}",      "—"),
            ("Anomaly Resilience",      f"{s.anomaly_resilience_score:.4f}", f"{a.anomaly_resilience_score:.4f}", "—"),
            ("Total Incidents",         str(s.total_incidents),             str(a.total_incidents),             "—"),
        ]
        for row in rows:
            table.add_row(*row)

        console.print(table)
        console.print(f"\n[bold]Run duration:[/bold] {self.run_duration_seconds:.2f}s")

    def to_dict(self) -> dict:
        return {
            "static": self.static_summary.to_dict(),
            "adaptive": self.adaptive_summary.to_dict(),
            "incident_reduction_pct": self.incident_reduction_pct,
            "override_reduction_pct": self.override_reduction_pct,
            "run_duration_seconds": round(self.run_duration_seconds, 2),
            "config": self.config,
        }


class ExperimentRunner:
    """
    Runs static and adaptive simulations with identical traffic/anomaly seeds
    so results are directly comparable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    def run_comparison(self, **override_params) -> ExperimentResult:
        """
        Run both static and adaptive simulations and return comparison results.
        Both runs use the same seed so anomaly injection is identical.
        """
        params = {**self._config, **override_params}

        sim_kwargs = {
            "total_steps": params.get("total_steps", 500),
            "decisions_per_step": params.get("decisions_per_step", 20),
            "base_failure_rate": params.get("base_failure_rate", 0.08),
            "anomaly_probability": params.get("anomaly_probability", 0.05),
            "anomaly_duration_steps": params.get("anomaly_duration_steps", 10),
            "peak_multiplier": params.get("peak_multiplier", 3.5),
            "alpha": params.get("alpha", 0.05),
            "beta": params.get("beta", 0.30),
            "safe_threshold": params.get("safe_threshold", 0.25),
            "risk_threshold": params.get("risk_threshold", 0.60),
            "initial_boundary": params.get("initial_boundary", 0.5),
            "min_boundary": params.get("min_boundary", 0.05),
            "max_boundary": params.get("max_boundary", 0.95),
            "window_size": params.get("window_size", 50),
            "headroom_buffer": params.get("headroom_buffer", 0.15),
            "min_update_interval_seconds": 0.0,  # disable rate limit in simulation
            "seed": params.get("seed", 42),
        }

        t_start = time.monotonic()

        # --- Static run ---
        logger.info("Running STATIC simulation...")
        static_sim = Simulator(
            **sim_kwargs,
            adaptive=False,
            audit_log_path="logs/static_audit.jsonl",
        )
        static_steps = static_sim.run()
        static_summary = static_sim.metrics_collector.summarize(label="static")

        # --- Adaptive run ---
        logger.info("Running ADAPTIVE simulation...")
        adaptive_sim = Simulator(
            **sim_kwargs,
            adaptive=True,
            audit_log_path="logs/adaptive_audit.jsonl",
        )
        adaptive_steps = adaptive_sim.run()
        adaptive_summary = adaptive_sim.metrics_collector.summarize(label="adaptive")

        run_duration = time.monotonic() - t_start

        return ExperimentResult(
            static_summary=static_summary,
            adaptive_summary=adaptive_summary,
            static_steps=static_steps,
            adaptive_steps=adaptive_steps,
            config=sim_kwargs,
            run_duration_seconds=run_duration,
        )

    def run_sensitivity_sweep(
        self,
        param_name: str,
        param_values: List[Any],
        **base_params,
    ) -> List[ExperimentResult]:
        """
        Sweep a single parameter across values, running comparison for each.
        Useful for calibrating α, β, thresholds, etc.
        """
        results = []
        for val in param_values:
            logger.info("Sensitivity sweep: %s = %s", param_name, val)
            result = self.run_comparison(**{**base_params, param_name: val})
            results.append(result)
        return results
