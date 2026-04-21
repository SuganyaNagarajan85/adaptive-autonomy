"""
Production Experiment Runner — scenario-driven static vs adaptive comparison.

Extends the existing ExperimentRunner with:
  - ScenarioConfig-driven simulation (replaces raw parameter dicts)
  - ProductionSimulator injection into Simulator
  - ScenarioExperimentResult with full scenario metadata
  - Multi-scenario batch runs
  - JSON result persistence

The runner ensures BOTH static and adaptive systems see IDENTICAL environment
conditions: same PatternGenerator seed, same traffic patterns, same anomaly timing.
Differences in outcomes are purely due to the presence/absence of the feedback loop.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.metrics import StepMetrics
from src.experiments.production_metrics import Metrics, ScenarioMetrics
from src.simulation.production_simulator import ProductionSimulator
from src.simulation.scenarios import ScenarioConfig, SCENARIO_REGISTRY, get_scenario
from src.simulation.simulator import Simulator

logger = logging.getLogger(__name__)


# ─── Result container ────────────────────────────────────────────────────────

@dataclass
class ScenarioExperimentResult:
    """
    Full result of a scenario-driven static vs adaptive comparison run.

    Contains:
      - Raw per-step metrics for both runs (for custom analysis)
      - Computed ScenarioMetrics for both runs
      - Scenario config (for reproducibility)
      - Run provenance (timing, seeds)
    """
    scenario: ScenarioConfig
    static_metrics: ScenarioMetrics
    adaptive_metrics: ScenarioMetrics
    static_steps: List[StepMetrics]
    adaptive_steps: List[StepMetrics]
    run_duration_seconds: float = 0.0

    # ── Derived comparison properties ────────────────────────────────────

    @property
    def incident_reduction_pct(self) -> float:
        """Reduction in mean incident rate: adaptive vs static (positive = better)."""
        s = self.static_metrics.mean_incident_rate
        a = self.adaptive_metrics.mean_incident_rate
        return round(((s - a) / max(s, 1e-6)) * 100, 2)

    @property
    def efficiency_gain(self) -> float:
        """Absolute gain in mean efficiency: adaptive − static."""
        return round(self.adaptive_metrics.mean_efficiency - self.static_metrics.mean_efficiency, 4)

    @property
    def boundary_flexibility(self) -> float:
        """
        How much the adaptive boundary moved vs the static (always 0).
        Higher = more responsive governance.
        """
        return round(self.adaptive_metrics.boundary_stability_stddev, 4)

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario.name,
            "scenario_description": self.scenario.description,
            "run_duration_seconds": round(self.run_duration_seconds, 2),
            "static": self.static_metrics.to_dict(),
            "adaptive": self.adaptive_metrics.to_dict(),
            "comparison": {
                "incident_reduction_pct": self.incident_reduction_pct,
                "efficiency_gain": self.efficiency_gain,
                "boundary_flexibility": self.boundary_flexibility,
            },
        }

    def save_json(self, path: str) -> str:
        """Persist results as pretty-printed JSON. Returns the path written."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        logger.info("Results saved to %s", p)
        return str(p)

    def print_comparison(self) -> None:
        """Rich-formatted comparison table to stdout."""
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
        except ImportError:
            print(self._plain_summary())
            return

        s = self.static_metrics
        a = self.adaptive_metrics

        table = Table(
            title=f"Scenario: {self.scenario.name.upper()} — Static vs Adaptive",
            show_header=True,
        )
        table.add_column("Metric", style="bold cyan")
        table.add_column("Static", justify="right")
        table.add_column("Adaptive", justify="right")
        table.add_column("Δ (adaptive − static)", justify="right", style="green")

        def row(name, sv, av, fmt=".4f"):
            delta = av - sv
            sign = "+" if delta >= 0 else ""
            table.add_row(name, f"{sv:{fmt}}", f"{av:{fmt}}", f"{sign}{delta:{fmt}}")

        row("Mean Incident Rate",      s.mean_incident_rate,        a.mean_incident_rate)
        row("Mean Override Rate",      s.mean_override_rate,        a.mean_override_rate)
        row("Mean Rollback Rate",      s.mean_rollback_rate,        a.mean_rollback_rate)
        row("Mean Autonomy Util.",     s.mean_autonomy_utilization, a.mean_autonomy_utilization)
        row("Mean Efficiency",         s.mean_efficiency,           a.mean_efficiency)
        row("Min Efficiency",          s.min_efficiency,            a.min_efficiency)
        row("Eff. During Anomaly",     s.efficiency_during_anomaly, a.efficiency_during_anomaly)
        row("Eff. During Normal",      s.efficiency_during_normal,  a.efficiency_during_normal)
        row("Boundary Stability σ",    s.boundary_stability_stddev, a.boundary_stability_stddev)
        row("Stability Score",         s.stability_score,           a.stability_score)
        row("IR Anomaly Period",           s.mean_incident_rate_anomaly,        a.mean_incident_rate_anomaly)
        row("Containment Score",           s.containment_score,                 a.containment_score)
        row("Decision-Driven Efficiency",  s.mean_decision_driven_efficiency,   a.mean_decision_driven_efficiency)
        row("DDE During Anomaly",          s.dde_during_anomaly,                a.dde_during_anomaly)
        row("DDE During Normal",           s.dde_during_normal,                 a.dde_during_normal)
        row("Autonomous Success Rate",     s.mean_autonomous_success_rate,      a.mean_autonomous_success_rate)
        table.add_row("CB Activations",    str(s.circuit_breaker_activations),  str(a.circuit_breaker_activations), "—")
        row("Final Mean Boundary",         s.final_mean_boundary,               a.final_mean_boundary)

        table.add_row(
            "Incident Reduction",
            "—", "—",
            f"{self.incident_reduction_pct:+.1f}%",
        )
        console.print(table)
        console.print(
            f"\n[bold]Scenario:[/bold] {self.scenario.description}\n"
            f"[bold]Run time:[/bold] {self.run_duration_seconds:.2f}s"
        )

    def _plain_summary(self) -> str:
        lines = [
            f"Scenario: {self.scenario.name}",
            f"  Incident reduction: {self.incident_reduction_pct:+.1f}%",
            f"  Efficiency gain:    {self.efficiency_gain:+.4f}",
            f"  Run time:           {self.run_duration_seconds:.2f}s",
        ]
        return "\n".join(lines)


# ─── Runner ──────────────────────────────────────────────────────────────────

class ProductionExperimentRunner:
    """
    Runs static vs adaptive comparison experiments driven by ScenarioConfig.

    Both runs use the SAME ProductionSimulator (same seed, same pattern) so
    environment conditions are identical — differences in results come purely
    from the presence or absence of the feedback loop.

    Usage::

        runner = ProductionExperimentRunner()

        # Single scenario
        result = runner.run_scenario(flash_sale_scenario())
        result.print_comparison()

        # All three canonical scenarios
        results = runner.run_all_scenarios()
    """

    def __init__(
        self,
        base_sim_kwargs: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs",
    ) -> None:
        """
        Args:
            base_sim_kwargs: extra kwargs forwarded to both Simulator instances.
                             Scenario values take priority over these.
            output_dir: base directory for JSON and plot outputs.
        """
        self._base_kwargs = base_sim_kwargs or {}
        self._output_dir = output_dir

    def run_scenario(
        self,
        scenario: ScenarioConfig,
        save_json: bool = True,
    ) -> ScenarioExperimentResult:
        """
        Run static and adaptive simulations under the given scenario.

        Both simulators share the same ProductionSimulator seed so they
        see identical traffic/error/latency curves.

        Args:
            scenario: ScenarioConfig (from scenarios.py factory functions)
            save_json: whether to persist results to JSON in output_dir

        Returns:
            ScenarioExperimentResult with all metrics
        """
        logger.info("Running scenario: %s (%d steps)", scenario.name, scenario.total_steps)

        sim_kwargs = self._build_sim_kwargs(scenario)
        t_start = time.monotonic()

        # ── Static run ────────────────────────────────────────────────
        logger.info("  [static]   starting...")
        static_prod_sim = ProductionSimulator(scenario)
        static_sim = Simulator(
            **sim_kwargs,
            adaptive=False,
            production_sim=static_prod_sim,
            audit_log_path=f"logs/{scenario.name}_static_audit.jsonl",
        )
        static_steps = static_sim.run()

        # ── Adaptive run ──────────────────────────────────────────────
        logger.info("  [adaptive] starting...")
        adaptive_prod_sim = ProductionSimulator(scenario)  # same config = same seed
        adaptive_sim = Simulator(
            **sim_kwargs,
            adaptive=True,
            production_sim=adaptive_prod_sim,
            audit_log_path=f"logs/{scenario.name}_adaptive_audit.jsonl",
        )
        adaptive_steps = adaptive_sim.run()

        run_duration = time.monotonic() - t_start

        # ── Compute metrics ───────────────────────────────────────────
        metrics_engine = Metrics(scenario_name=scenario.name)

        # Compute static first to get anomaly IR reference for containment score
        static_m = metrics_engine.compute(static_steps, label="static")
        adaptive_m = metrics_engine.compute(
            adaptive_steps,
            label="adaptive",
            containment_reference_ir=static_m.mean_incident_rate_anomaly,
        )

        result = ScenarioExperimentResult(
            scenario=scenario,
            static_metrics=static_m,
            adaptive_metrics=adaptive_m,
            static_steps=static_steps,
            adaptive_steps=adaptive_steps,
            run_duration_seconds=run_duration,
        )

        if save_json:
            json_path = f"{self._output_dir}/{scenario.name}/results.json"
            result.save_json(json_path)

        logger.info(
            "Scenario '%s' complete in %.2fs | incident reduction: %+.1f%%",
            scenario.name, run_duration, result.incident_reduction_pct,
        )
        return result

    def run_all_scenarios(
        self,
        scenario_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        save_json: bool = True,
    ) -> Dict[str, ScenarioExperimentResult]:
        """
        Run all three canonical scenarios (flash_sale, degradation, recovery).

        Args:
            scenario_kwargs: per-scenario override dicts, keyed by scenario name.
            save_json: persist results for each scenario.

        Returns:
            Dict mapping scenario name → ScenarioExperimentResult
        """
        scenario_kwargs = scenario_kwargs or {}
        results: Dict[str, ScenarioExperimentResult] = {}

        for name, factory in SCENARIO_REGISTRY.items():
            kwargs = scenario_kwargs.get(name, {})
            scenario = factory(**kwargs)
            results[name] = self.run_scenario(scenario, save_json=save_json)

        return results

    def run_named_scenario(
        self,
        name: str,
        save_json: bool = True,
        **kwargs,
    ) -> ScenarioExperimentResult:
        """Convenience: look up by name and run."""
        return self.run_scenario(get_scenario(name, **kwargs), save_json=save_json)

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_sim_kwargs(self, scenario: ScenarioConfig) -> Dict[str, Any]:
        """
        Merge base_sim_kwargs with scenario-derived values.
        Scenario values win over base_sim_kwargs.
        """
        base = {
            "total_steps": scenario.total_steps,
            "decisions_per_step": scenario.decisions_per_step,
            "initial_boundary": scenario.initial_boundary,
            "alpha": scenario.alpha,
            "beta": scenario.beta,
            "safe_threshold": scenario.safe_threshold,
            "risk_threshold": scenario.risk_threshold,
            "window_size": scenario.window_size,
            "headroom_buffer": scenario.headroom_buffer,
            "seed": scenario.seed,
            "min_update_interval_seconds": 0.0,   # no rate limiting in simulation
        }
        # base_sim_kwargs can add fields not in ScenarioConfig (e.g. shadow_mode)
        merged = {**self._base_kwargs, **base}
        return merged
