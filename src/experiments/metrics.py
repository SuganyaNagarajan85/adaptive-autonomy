"""
Metrics Collector — captures per-step simulation metrics for experiment comparison.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StepMetrics:
    """All metrics captured for a single simulation step."""
    step: int
    is_anomaly: bool
    anomaly_type: str
    total_decisions: int
    auto_decisions: int
    human_review_decisions: int
    incidents: int
    rollbacks: int
    overrides: int
    boundary_snapshots: Dict[str, float]  # decision_type → B at step start
    window_metrics: Dict[str, dict]        # decision_type → WindowMetrics.to_dict()
    review_queue_depth: int
    successful_auto_decisions: int = 0    # auto decisions with no adverse outcome
    circuit_breaker_fired: bool = False   # True if spike CB triggered this step

    @property
    def incident_rate(self) -> float:
        return self.incidents / max(self.total_decisions, 1)

    @property
    def override_rate(self) -> float:
        return self.overrides / max(self.total_decisions, 1)

    @property
    def rollback_rate(self) -> float:
        return self.rollbacks / max(self.total_decisions, 1)

    @property
    def autonomy_utilization(self) -> float:
        return self.auto_decisions / max(self.total_decisions, 1)

    @property
    def decision_driven_efficiency(self) -> float:
        """Fraction of ALL decisions handled autonomously AND correctly.
        Since human-review decisions always resolve SUCCESS in simulation,
        all adverse outcomes (incidents/overrides/rollbacks) come from
        autonomous decisions only — so this is exact, not an approximation.
        """
        return self.successful_auto_decisions / max(self.total_decisions, 1)

    @property
    def autonomous_success_rate(self) -> float:
        """Of decisions that went autonomous, what fraction succeeded?"""
        return self.successful_auto_decisions / max(self.auto_decisions, 1)

    @property
    def mean_boundary(self) -> float:
        if not self.boundary_snapshots:
            return 0.0
        return sum(self.boundary_snapshots.values()) / len(self.boundary_snapshots)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "total_decisions": self.total_decisions,
            "auto_decisions": self.auto_decisions,
            "human_review_decisions": self.human_review_decisions,
            "incidents": self.incidents,
            "rollbacks": self.rollbacks,
            "overrides": self.overrides,
            "incident_rate": round(self.incident_rate, 4),
            "override_rate": round(self.override_rate, 4),
            "rollback_rate": round(self.rollback_rate, 4),
            "autonomy_utilization": round(self.autonomy_utilization, 4),
            "mean_boundary": round(self.mean_boundary, 4),
            "review_queue_depth": self.review_queue_depth,
            "boundary_snapshots": {k: round(v, 4) for k, v in self.boundary_snapshots.items()},
            "successful_auto_decisions": self.successful_auto_decisions,
            "decision_driven_efficiency": round(self.decision_driven_efficiency, 4),
            "autonomous_success_rate": round(self.autonomous_success_rate, 4),
            "circuit_breaker_fired": self.circuit_breaker_fired,
        }


@dataclass
class ExperimentSummary:
    """Aggregate statistics across all steps of one experiment run."""
    label: str
    total_steps: int
    total_decisions: int
    total_incidents: int
    total_rollbacks: int
    total_overrides: int
    mean_incident_rate: float
    mean_override_rate: float
    mean_rollback_rate: float
    mean_autonomy_utilization: float
    final_mean_boundary: float
    boundary_stability: float        # stddev of mean_boundary — lower = more stable
    anomaly_resilience_score: float  # how well system recovered after anomalies

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "total_steps": self.total_steps,
            "total_decisions": self.total_decisions,
            "total_incidents": self.total_incidents,
            "total_rollbacks": self.total_rollbacks,
            "total_overrides": self.total_overrides,
            "mean_incident_rate": round(self.mean_incident_rate, 4),
            "mean_override_rate": round(self.mean_override_rate, 4),
            "mean_rollback_rate": round(self.mean_rollback_rate, 4),
            "mean_autonomy_utilization": round(self.mean_autonomy_utilization, 4),
            "final_mean_boundary": round(self.final_mean_boundary, 4),
            "boundary_stability_stddev": round(self.boundary_stability, 4),
            "anomaly_resilience_score": round(self.anomaly_resilience_score, 4),
        }


class MetricsCollector:
    """Accumulates StepMetrics across simulation steps."""

    def __init__(self) -> None:
        self._steps: List[StepMetrics] = []

    def record(self, metrics: StepMetrics) -> None:
        self._steps.append(metrics)

    def all_steps(self) -> List[StepMetrics]:
        return list(self._steps)

    def summarize(self, label: str = "experiment") -> ExperimentSummary:
        steps = self._steps
        if not steps:
            raise ValueError("No steps recorded")

        total_decisions = sum(s.total_decisions for s in steps)
        total_incidents = sum(s.incidents for s in steps)
        total_rollbacks = sum(s.rollbacks for s in steps)
        total_overrides = sum(s.overrides for s in steps)

        incident_rates = [s.incident_rate for s in steps]
        override_rates = [s.override_rate for s in steps]
        rollback_rates = [s.rollback_rate for s in steps]
        auto_utils = [s.autonomy_utilization for s in steps]
        mean_boundaries = [s.mean_boundary for s in steps]

        # Anomaly resilience: ratio of post-anomaly incident rate vs pre-anomaly
        anomaly_steps = [s for s in steps if s.is_anomaly]
        normal_steps = [s for s in steps if not s.is_anomaly]
        if anomaly_steps and normal_steps:
            anomaly_ir = statistics.mean(s.incident_rate for s in anomaly_steps)
            normal_ir = statistics.mean(s.incident_rate for s in normal_steps)
            # Score 1.0 = perfect (same rate); lower = adaptive system contained risk better
            resilience = 1.0 - min(1.0, max(0.0, (anomaly_ir - normal_ir)))
        else:
            resilience = 1.0

        return ExperimentSummary(
            label=label,
            total_steps=len(steps),
            total_decisions=total_decisions,
            total_incidents=total_incidents,
            total_rollbacks=total_rollbacks,
            total_overrides=total_overrides,
            mean_incident_rate=statistics.mean(incident_rates),
            mean_override_rate=statistics.mean(override_rates),
            mean_rollback_rate=statistics.mean(rollback_rates),
            mean_autonomy_utilization=statistics.mean(auto_utils),
            final_mean_boundary=mean_boundaries[-1] if mean_boundaries else 0.0,
            boundary_stability=statistics.stdev(mean_boundaries) if len(mean_boundaries) > 1 else 0.0,
            anomaly_resilience_score=resilience,
        )

    def get_series(self, field_name: str) -> List[float]:
        """Extract a time series from step metrics by field name."""
        return [getattr(s, field_name, 0.0) for s in self._steps]

    def get_boundary_series(self, decision_type: str) -> List[float]:
        return [s.boundary_snapshots.get(decision_type, 0.0) for s in self._steps]

    def get_anomaly_flags(self) -> List[bool]:
        return [s.is_anomaly for s in self._steps]
