"""
Production Metrics — scenario-aware metric computation for experiment runs.

Extends the existing ExperimentSummary with production-specific measures:

  efficiency  = autonomy_utilization − incident_rate
                High efficiency means: high automation AND low incidents.
                A static system with B=1.0 but poor incident control has low efficiency.

  stability   = 1 − boundary_stability_stddev (normalized)
                How stable was the boundary over time? High = consistent governance.

  recovery_speed (recovery scenario only):
                Steps from peak error until B returned to a defined target level.

  containment_score (flash_sale / degradation):
                How well did the adaptive system suppress incident_rate during the
                anomaly window vs the static system?

All per-step metrics build on StepMetrics from src/experiments/metrics.py —
no duplication, only extension.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.experiments.metrics import StepMetrics


# ─── Per-step efficiency ──────────────────────────────────────────────────────

@dataclass
class StepEfficiency:
    """Efficiency decomposition for a single simulation step."""
    step: int
    autonomy_utilization: float
    incident_rate: float
    efficiency: float           # = autonomy_utilization − incident_rate
    mean_boundary: float
    is_anomaly: bool
    anomaly_type: str

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "autonomy_utilization": round(self.autonomy_utilization, 4),
            "incident_rate": round(self.incident_rate, 4),
            "efficiency": round(self.efficiency, 4),
            "mean_boundary": round(self.mean_boundary, 4),
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
        }


# ─── Aggregate metrics ────────────────────────────────────────────────────────

@dataclass
class ScenarioMetrics:
    """
    Full metric set for one (static or adaptive) run under a named scenario.

    All rates are per-decision fractions unless stated otherwise.
    """
    label: str                   # "static" | "adaptive"
    scenario_name: str
    total_steps: int
    total_decisions: int

    # Safety metrics
    mean_incident_rate: float
    mean_override_rate: float
    mean_rollback_rate: float
    total_incidents: int
    total_overrides: int
    total_rollbacks: int

    # Autonomy metrics
    mean_autonomy_utilization: float
    final_mean_boundary: float
    initial_mean_boundary: float

    # Efficiency  (autonomy_utilization − incident_rate)
    mean_efficiency: float
    min_efficiency: float           # worst single-step efficiency
    efficiency_during_anomaly: float
    efficiency_during_normal: float

    # Stability (boundary variance)
    boundary_stability_stddev: float
    stability_score: float          # 1 − normalised stddev ∈ [0,1]

    # Anomaly-period performance
    anomaly_steps: int
    mean_incident_rate_anomaly: float
    mean_incident_rate_normal: float
    containment_score: float        # 1 − (anomaly_ir / static_ref_ir), set by runner

    # Decision-driven efficiency
    mean_decision_driven_efficiency: float   # successful_auto / total decisions
    dde_during_anomaly: float
    dde_during_normal: float
    mean_autonomous_success_rate: float      # successful_auto / auto_decisions
    circuit_breaker_activations: int         # steps where CB fired

    # Per-step efficiency series (for plotting)
    step_efficiencies: List[StepEfficiency] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "scenario_name": self.scenario_name,
            "total_steps": self.total_steps,
            "total_decisions": self.total_decisions,
            "mean_incident_rate": round(self.mean_incident_rate, 4),
            "mean_override_rate": round(self.mean_override_rate, 4),
            "mean_rollback_rate": round(self.mean_rollback_rate, 4),
            "total_incidents": self.total_incidents,
            "total_overrides": self.total_overrides,
            "total_rollbacks": self.total_rollbacks,
            "mean_autonomy_utilization": round(self.mean_autonomy_utilization, 4),
            "final_mean_boundary": round(self.final_mean_boundary, 4),
            "initial_mean_boundary": round(self.initial_mean_boundary, 4),
            "mean_efficiency": round(self.mean_efficiency, 4),
            "min_efficiency": round(self.min_efficiency, 4),
            "efficiency_during_anomaly": round(self.efficiency_during_anomaly, 4),
            "efficiency_during_normal": round(self.efficiency_during_normal, 4),
            "boundary_stability_stddev": round(self.boundary_stability_stddev, 4),
            "stability_score": round(self.stability_score, 4),
            "anomaly_steps": self.anomaly_steps,
            "mean_incident_rate_anomaly": round(self.mean_incident_rate_anomaly, 4),
            "mean_incident_rate_normal": round(self.mean_incident_rate_normal, 4),
            "containment_score": round(self.containment_score, 4),
            "mean_decision_driven_efficiency": round(self.mean_decision_driven_efficiency, 4),
            "dde_during_anomaly": round(self.dde_during_anomaly, 4),
            "dde_during_normal": round(self.dde_during_normal, 4),
            "mean_autonomous_success_rate": round(self.mean_autonomous_success_rate, 4),
            "circuit_breaker_activations": self.circuit_breaker_activations,
        }


# ─── Metrics computer ────────────────────────────────────────────────────────

class Metrics:
    """
    Computes ScenarioMetrics from a list of StepMetrics.

    Designed to be called independently for both the static and adaptive runs
    so the runner can compare them side by side.

    Usage::

        metrics = Metrics(scenario_name="flash_sale")
        static_m  = metrics.compute(static_steps,  label="static")
        adaptive_m = metrics.compute(adaptive_steps, label="adaptive")
    """

    def __init__(self, scenario_name: str = "unknown") -> None:
        self._scenario_name = scenario_name

    def compute(
        self,
        steps: List[StepMetrics],
        label: str = "run",
        containment_reference_ir: Optional[float] = None,
    ) -> ScenarioMetrics:
        """
        Compute the full ScenarioMetrics for one simulation run.

        Args:
            steps: per-step metrics from MetricsCollector.all_steps()
            label: "static" or "adaptive"
            containment_reference_ir: the static run's anomaly incident rate,
                used to compute containment_score for the adaptive run.
                Pass None when computing the static run itself.

        Returns:
            ScenarioMetrics with all fields populated.
        """
        if not steps:
            raise ValueError("steps list is empty — nothing to compute")

        total_decisions = sum(s.total_decisions for s in steps)
        anomaly_steps = [s for s in steps if s.is_anomaly]
        normal_steps  = [s for s in steps if not s.is_anomaly]

        # ── Safety rates ────────────────────────────────────────────────
        ir_series  = [s.incident_rate        for s in steps]
        or_series  = [s.override_rate        for s in steps]
        rb_series  = [s.rollback_rate        for s in steps]
        au_series  = [s.autonomy_utilization for s in steps]
        mb_series  = [s.mean_boundary        for s in steps]

        mean_ir = statistics.mean(ir_series)
        mean_or = statistics.mean(or_series)
        mean_rb = statistics.mean(rb_series)
        mean_au = statistics.mean(au_series)

        # ── Efficiency per step ─────────────────────────────────────────
        step_efficiencies = [
            StepEfficiency(
                step=s.step,
                autonomy_utilization=s.autonomy_utilization,
                incident_rate=s.incident_rate,
                efficiency=s.autonomy_utilization - s.incident_rate,
                mean_boundary=s.mean_boundary,
                is_anomaly=s.is_anomaly,
                anomaly_type=s.anomaly_type,
            )
            for s in steps
        ]
        eff_series = [se.efficiency for se in step_efficiencies]
        mean_eff   = statistics.mean(eff_series)
        min_eff    = min(eff_series)

        eff_anomaly = statistics.mean(
            [se.efficiency for se in step_efficiencies if se.is_anomaly]
        ) if anomaly_steps else mean_eff

        eff_normal = statistics.mean(
            [se.efficiency for se in step_efficiencies if not se.is_anomaly]
        ) if normal_steps else mean_eff

        # ── Stability ───────────────────────────────────────────────────
        b_std = statistics.stdev(mb_series) if len(mb_series) > 1 else 0.0
        # Normalize: max theoretical stddev for B ∈ [0,1] is 0.5
        stability_score = max(0.0, 1.0 - (b_std / 0.5))

        # ── Anomaly-period incident rates ───────────────────────────────
        ir_anomaly = statistics.mean(s.incident_rate for s in anomaly_steps) if anomaly_steps else 0.0
        ir_normal  = statistics.mean(s.incident_rate for s in normal_steps)  if normal_steps  else 0.0

        # ── Containment score ───────────────────────────────────────────
        # How much better is this run vs the reference (static) during anomalies?
        if containment_reference_ir is not None and containment_reference_ir > 0:
            containment = 1.0 - min(1.0, ir_anomaly / containment_reference_ir)
        else:
            containment = 0.0

        # ── Decision-driven efficiency ──────────────────────────────────
        dde_series  = [s.decision_driven_efficiency for s in steps]
        asr_series  = [s.autonomous_success_rate    for s in steps]
        mean_dde    = statistics.mean(dde_series)
        mean_asr    = statistics.mean(asr_series)

        dde_anomaly = statistics.mean(
            s.decision_driven_efficiency for s in anomaly_steps
        ) if anomaly_steps else mean_dde

        dde_normal  = statistics.mean(
            s.decision_driven_efficiency for s in normal_steps
        ) if normal_steps else mean_dde

        cb_activations = sum(1 for s in steps if s.circuit_breaker_fired)

        return ScenarioMetrics(
            label=label,
            scenario_name=self._scenario_name,
            total_steps=len(steps),
            total_decisions=total_decisions,
            mean_incident_rate=mean_ir,
            mean_override_rate=mean_or,
            mean_rollback_rate=mean_rb,
            total_incidents=sum(s.incidents for s in steps),
            total_overrides=sum(s.overrides for s in steps),
            total_rollbacks=sum(s.rollbacks for s in steps),
            mean_autonomy_utilization=mean_au,
            final_mean_boundary=mb_series[-1],
            initial_mean_boundary=mb_series[0],
            mean_efficiency=mean_eff,
            min_efficiency=min_eff,
            efficiency_during_anomaly=eff_anomaly,
            efficiency_during_normal=eff_normal,
            boundary_stability_stddev=b_std,
            stability_score=stability_score,
            anomaly_steps=len(anomaly_steps),
            mean_incident_rate_anomaly=ir_anomaly,
            mean_incident_rate_normal=ir_normal,
            containment_score=containment,
            mean_decision_driven_efficiency=mean_dde,
            dde_during_anomaly=dde_anomaly,
            dde_during_normal=dde_normal,
            mean_autonomous_success_rate=mean_asr,
            circuit_breaker_activations=cb_activations,
            step_efficiencies=step_efficiencies,
        )

    def compute_per_step(self, steps: List[StepMetrics]) -> List[StepEfficiency]:
        """Return just the per-step efficiency series without full metric computation."""
        return [
            StepEfficiency(
                step=s.step,
                autonomy_utilization=s.autonomy_utilization,
                incident_rate=s.incident_rate,
                efficiency=s.autonomy_utilization - s.incident_rate,
                mean_boundary=s.mean_boundary,
                is_anomaly=s.is_anomaly,
                anomaly_type=s.anomaly_type,
            )
            for s in steps
        ]
