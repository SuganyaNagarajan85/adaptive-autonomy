"""
Decision Engine Models — core domain objects for the Autonomy Learning Loop.
All decisions in the e-commerce platform flow through these types.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """E-commerce decision classes. Each class maintains its own autonomy boundary B."""
    PRODUCT_RECOMMENDATION = "PRODUCT_RECOMMENDATION"  # ranking/recs engine
    OFFER_SELECTION = "OFFER_SELECTION"                # coupon / discount vending
    FRAUD_DETECTION = "FRAUD_DETECTION"                # risk scoring on transactions
    PRICING = "PRICING"                                # dynamic price computation
    NOTIFICATION = "NOTIFICATION"                      # push / email trigger
    SEARCH_RANKING = "SEARCH_RANKING"                  # search result ordering


class ExecutionMode(str, Enum):
    """Whether the decision was executed autonomously or routed to human review."""
    AUTO = "AUTO"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    SHADOW = "SHADOW"   # executed autonomously but also queued for human comparison


class DecisionOutcome(str, Enum):
    """Observed outcome after the decision was acted upon."""
    SUCCESS = "SUCCESS"
    ROLLBACK = "ROLLBACK"      # decision was reversed / undone
    OVERRIDE = "OVERRIDE"      # human changed the decision post-execution
    INCIDENT = "INCIDENT"      # decision correlated with a service incident
    PENDING = "PENDING"        # outcome not yet observed (async)


class DecisionContext(BaseModel):
    """Contextual metadata attached to a decision for attribution-tolerant analysis."""
    model_config = {"protected_namespaces": ()}

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_latency_ms: Optional[float] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_version: str = "v1.0"
    is_anomaly_period: bool = False
    traffic_multiplier: float = 1.0     # 1.0 = normal; >1 = flash sale / spike
    extra: Dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """A single decision request flowing through the system."""
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: DecisionType
    context: DecisionContext
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    autonomy_boundary_at_decision: float = Field(
        ge=0.0, le=1.0, description="Snapshot of B when this decision was routed"
    )
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    outcome_timestamp: Optional[datetime] = None

    def resolve_outcome(self, outcome: DecisionOutcome) -> None:
        self.outcome = outcome
        self.outcome_timestamp = datetime.utcnow()

    @property
    def is_autonomous(self) -> bool:
        return self.execution_mode in (ExecutionMode.AUTO, ExecutionMode.SHADOW)

    @property
    def is_adverse(self) -> bool:
        return self.outcome in (
            DecisionOutcome.ROLLBACK,
            DecisionOutcome.OVERRIDE,
            DecisionOutcome.INCIDENT,
        )
