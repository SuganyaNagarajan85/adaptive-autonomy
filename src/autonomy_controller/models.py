"""
Autonomy Controller Models — state and routing output types.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.decision_engine.models import DecisionType, ExecutionMode


class AutonomyState(BaseModel):
    """
    Per-decision-class autonomy boundary state.
    B ∈ [0, 1] — higher = more autonomous, lower = more human oversight.
    """
    decision_type: DecisionType
    boundary: float = Field(ge=0.0, le=1.0, description="Current autonomy score B")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    update_count: int = 0
    consecutive_expansions: int = 0   # track oscillation risk
    consecutive_contractions: int = 0
    frozen: bool = False              # safety freeze flag — disables expansion temporarily

    def snapshot(self) -> Dict:
        return {
            "decision_type": self.decision_type.value,
            "boundary": round(self.boundary, 4),
            "last_updated": self.last_updated.isoformat(),
            "update_count": self.update_count,
            "frozen": self.frozen,
        }


class RoutingOutcome(str, Enum):
    AUTO_EXECUTE = "AUTO_EXECUTE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SHADOW_EXECUTE = "SHADOW_EXECUTE"   # auto + queued for human comparison


class RoutingDecision(BaseModel):
    """Output of AutonomyController.route() for a single incoming decision request."""
    decision_type: DecisionType
    boundary_at_routing: float
    outcome: RoutingOutcome
    execution_mode: ExecutionMode
    random_draw: float              # the U(0,1) draw — useful for audit / reproducibility
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: Optional[str] = None    # human-readable explanation
