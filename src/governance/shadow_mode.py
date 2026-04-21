"""
Shadow Mode Engine — runs decisions through both auto-execution and human judgment,
then compares outcomes to build a calibration dataset.

Shadow mode is the safest way to evaluate autonomy readiness before raising
the boundary B. It provides an empirical ground-truth signal: how often does
the autonomous path agree with human judgment?

Paper §3.4: "evaluated through simulation, shadow-mode deployment, and phased rollout"
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ShadowComparison:
    """One shadow mode comparison record."""
    decision_id: str
    decision_type: str
    auto_outcome: str
    human_outcome: str
    agreement: bool
    confidence_score: float
    boundary_at_decision: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "auto_outcome": self.auto_outcome,
            "human_outcome": self.human_outcome,
            "agreement": self.agreement,
            "confidence_score": round(self.confidence_score, 4),
            "boundary_at_decision": round(self.boundary_at_decision, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class ShadowModeEngine:
    """
    Tracks shadow mode comparisons and computes agreement rates per decision class.

    Agreement rate can feed back into the boundary update as a positive trust signal:
    high agreement → evidence for expansion readiness.
    """

    def __init__(self) -> None:
        self._comparisons: List[ShadowComparison] = []
        self._lock = threading.Lock()

    def record(
        self,
        decision_id: str,
        decision_type: str,
        auto_outcome: str,
        human_outcome: str,
        confidence_score: float,
        boundary_at_decision: float,
    ) -> ShadowComparison:
        comparison = ShadowComparison(
            decision_id=decision_id,
            decision_type=decision_type,
            auto_outcome=auto_outcome,
            human_outcome=human_outcome,
            agreement=(auto_outcome == human_outcome),
            confidence_score=confidence_score,
            boundary_at_decision=boundary_at_decision,
        )
        with self._lock:
            self._comparisons.append(comparison)
        return comparison

    def agreement_rate(self, decision_type: Optional[str] = None) -> float:
        """Overall or per-class agreement rate between auto and human outcomes."""
        with self._lock:
            items = (
                [c for c in self._comparisons if c.decision_type == decision_type]
                if decision_type
                else list(self._comparisons)
            )
        if not items:
            return 0.0
        return sum(1 for c in items if c.agreement) / len(items)

    def per_class_agreement(self) -> Dict[str, float]:
        with self._lock:
            types = {c.decision_type for c in self._comparisons}
        return {dt: self.agreement_rate(dt) for dt in types}

    def recent_comparisons(self, n: int = 20) -> List[dict]:
        with self._lock:
            return [c.to_dict() for c in self._comparisons[-n:]]

    def summary(self) -> dict:
        with self._lock:
            total = len(self._comparisons)
        return {
            "total_comparisons": total,
            "overall_agreement_rate": round(self.agreement_rate(), 4),
            "per_class": {k: round(v, 4) for k, v in self.per_class_agreement().items()},
        }
