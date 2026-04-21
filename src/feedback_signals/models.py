"""
Feedback Signal Models — operational consequence signals captured asynchronously
from the production decision system. These are the inputs to the Autonomy Learning Loop.

Based on Table 3 from the paper: rollback frequency, recovery cost, incident correlation,
human override rate, confidence-outcome divergence, trust indicators.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """
    Categories of operational consequence signals (paper §4.1, Table 3).
    Each type maps to a specific governance risk indicator.
    """
    ROLLBACK = "ROLLBACK"                    # decision was manually reversed
    OVERRIDE = "OVERRIDE"                    # human changed post-execution
    INCIDENT = "INCIDENT"                    # temporal correlation with incident
    TRUST_DEGRADATION = "TRUST_DEGRADATION"  # cart abandon / opt-out / complaint
    CONFIDENCE_DIVERGENCE = "CONFIDENCE_DIVERGENCE"  # high confidence, bad outcome


class SignalSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Severity → numeric weight multiplier applied on top of signal_type base weight
SEVERITY_MULTIPLIERS = {
    SignalSeverity.LOW: 0.5,
    SignalSeverity.MEDIUM: 1.0,
    SignalSeverity.HIGH: 1.5,
    SignalSeverity.CRITICAL: 2.5,
}


class FeedbackSignal(BaseModel):
    """
    A single asynchronous operational feedback event. Signals are produced
    off the hot path and queued for windowed aggregation in the control plane.
    """
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: SignalType
    severity: SignalSeverity = SignalSeverity.MEDIUM
    decision_id: Optional[str] = None         # linked decision (best-effort attribution)
    decision_type: Optional[str] = None       # decision class for per-class aggregation
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: str = ""
    recovery_cost_seconds: Optional[float] = None  # for ROLLBACK signals
    confidence_at_decision: Optional[float] = None  # for CONFIDENCE_DIVERGENCE

    @property
    def weighted_impact(self) -> float:
        """Pre-multiplied impact used by aggregator before applying type weights."""
        return SEVERITY_MULTIPLIERS[self.severity]
