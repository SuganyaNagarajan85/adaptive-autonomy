"""
Feedback Aggregator — window-based aggregation of operational feedback signals.

Implements §4.2 scalability architecture:
  - Signals aggregated at decision-class × temporal-window level
  - Attribution-tolerant: correlation-based, not causal
  - Produces a normalized risk score S_t ∈ [0, 1] per decision class per window

S_t formula:
    raw_score = Σ (signal_type_weight × severity_multiplier) for each signal in window
    S_t = min(1.0, raw_score / normalization_factor)

The normalization factor is empirically set so that "expected normal operation"
produces S_t ≈ 0.1 and a fully adverse window produces S_t → 1.0.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.feedback_signals.models import (
    FeedbackSignal,
    SignalType,
    SEVERITY_MULTIPLIERS,
)

logger = logging.getLogger(__name__)

# Default signal type → base weight mapping (paper §4.1 / Table 3 inspired)
DEFAULT_SIGNAL_WEIGHTS: Dict[SignalType, float] = {
    SignalType.ROLLBACK: 0.35,
    SignalType.OVERRIDE: 0.25,
    SignalType.INCIDENT: 0.30,
    SignalType.TRUST_DEGRADATION: 0.05,
    SignalType.CONFIDENCE_DIVERGENCE: 0.05,
}

# Normalization denominator: raw score accumulated if all window decisions had
# a MEDIUM severity INCIDENT (worst "expected" case without full meltdown).
# Adjust this to calibrate S_t sensitivity.
_NORM_FACTOR = 5.0


@dataclass
class WindowMetrics:
    """Aggregated risk metrics for one decision class over one time window."""
    decision_type: str
    window_index: int
    signal_count: int = 0
    signal_counts_by_type: Dict[str, int] = field(default_factory=dict)
    raw_risk_score: float = 0.0
    normalized_risk_score: float = 0.0   # S_t ∈ [0, 1]
    total_decisions: int = 0
    auto_decisions: int = 0
    human_review_decisions: int = 0
    adverse_outcomes: int = 0
    override_rate: float = 0.0
    rollback_rate: float = 0.0
    incident_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def autonomy_utilization(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return self.auto_decisions / self.total_decisions

    def to_dict(self) -> dict:
        return {
            "decision_type": self.decision_type,
            "window_index": self.window_index,
            "S_t": round(self.normalized_risk_score, 4),
            "signal_count": self.signal_count,
            "total_decisions": self.total_decisions,
            "auto_decisions": self.auto_decisions,
            "human_review_decisions": self.human_review_decisions,
            "adverse_outcomes": self.adverse_outcomes,
            "override_rate": round(self.override_rate, 4),
            "rollback_rate": round(self.rollback_rate, 4),
            "incident_rate": round(self.incident_rate, 4),
            "autonomy_utilization": round(self.autonomy_utilization, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class FeedbackAggregator:
    """
    Consumes raw FeedbackSignal events and produces per-class WindowMetrics.

    The aggregator operates independently from the request path.
    It is called by the Simulator at each window boundary.
    """

    def __init__(
        self,
        window_size: int = 50,
        signal_weights: Optional[Dict[SignalType, float]] = None,
        norm_factor: float = _NORM_FACTOR,
    ) -> None:
        self._window_size = window_size
        self._signal_weights = signal_weights or DEFAULT_SIGNAL_WEIGHTS
        self._norm_factor = norm_factor
        self._window_index = 0

        # Per-class accumulators: reset every window
        self._signal_buffer: Dict[str, List[FeedbackSignal]] = defaultdict(list)
        # Per-class decision counters
        self._decision_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "auto": 0, "human_review": 0, "adverse": 0,
                     "rollback": 0, "override": 0, "incident": 0}
        )

    def record_decision(
        self,
        decision_type: str,
        is_auto: bool,
        outcome: str,
    ) -> None:
        """
        Record each decision's routing and outcome for rate computation.
        Called synchronously as decisions resolve.
        """
        c = self._decision_counts[decision_type]
        c["total"] += 1
        if is_auto:
            c["auto"] += 1
        else:
            c["human_review"] += 1
        if outcome in ("ROLLBACK", "OVERRIDE", "INCIDENT"):
            c["adverse"] += 1
        if outcome == "ROLLBACK":
            c["rollback"] += 1
        elif outcome == "OVERRIDE":
            c["override"] += 1
        elif outcome == "INCIDENT":
            c["incident"] += 1

    def ingest_signal(self, signal: FeedbackSignal) -> None:
        """Ingest a single feedback signal into the current window buffer."""
        key = signal.decision_type or "UNKNOWN"
        self._signal_buffer[key].append(signal)

    def ingest_batch(self, signals: List[FeedbackSignal]) -> None:
        for s in signals:
            self.ingest_signal(s)

    def evaluate_window(self) -> Dict[str, WindowMetrics]:
        """
        Close the current window and produce WindowMetrics per decision class.
        Resets buffers and increments window counter.
        Called by the Simulator at each window boundary.
        """
        results: Dict[str, WindowMetrics] = {}

        all_types = set(self._signal_buffer.keys()) | set(self._decision_counts.keys())

        for dt in all_types:
            signals = self._signal_buffer.get(dt, [])
            counts = self._decision_counts.get(dt, {
                "total": 0, "auto": 0, "human_review": 0, "adverse": 0,
                "rollback": 0, "override": 0, "incident": 0
            })

            raw_score = self._compute_raw_score(signals)
            s_t = min(1.0, raw_score / self._norm_factor) if self._norm_factor > 0 else 0.0

            total = counts["total"]
            override_rate = counts["override"] / total if total > 0 else 0.0
            rollback_rate = counts["rollback"] / total if total > 0 else 0.0
            incident_rate = counts["incident"] / total if total > 0 else 0.0

            # Aggregate signal counts by type
            sig_by_type: Dict[str, int] = defaultdict(int)
            for s in signals:
                sig_by_type[s.signal_type.value] += 1

            metrics = WindowMetrics(
                decision_type=dt,
                window_index=self._window_index,
                signal_count=len(signals),
                signal_counts_by_type=dict(sig_by_type),
                raw_risk_score=round(raw_score, 4),
                normalized_risk_score=round(s_t, 4),
                total_decisions=total,
                auto_decisions=counts["auto"],
                human_review_decisions=counts["human_review"],
                adverse_outcomes=counts["adverse"],
                override_rate=round(override_rate, 4),
                rollback_rate=round(rollback_rate, 4),
                incident_rate=round(incident_rate, 4),
            )
            results[dt] = metrics

        # Reset for next window
        self._signal_buffer.clear()
        self._decision_counts.clear()
        self._window_index += 1

        logger.debug("Window %d evaluated: %d decision classes", self._window_index - 1, len(results))
        return results

    def _compute_raw_score(self, signals: List[FeedbackSignal]) -> float:
        """
        Weighted sum of feedback signals for one decision class in one window.

        raw_score = Σ [ type_weight × severity_multiplier ]
        """
        score = 0.0
        for signal in signals:
            type_weight = self._signal_weights.get(signal.signal_type, 0.1)
            severity_mult = SEVERITY_MULTIPLIERS[signal.severity]
            score += type_weight * severity_mult
        return score
