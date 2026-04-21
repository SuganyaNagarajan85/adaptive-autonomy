"""
Decision Engine — simulates a rule-based / ML-mock decision system.
In a real ecommerce system this would wrap your recommendation model,
pricing engine, fraud scorer, etc. Here we produce realistic mock decisions
with configurable confidence distributions per decision type.
"""
from __future__ import annotations

import random
from typing import Dict, Optional

from .models import Decision, DecisionContext, DecisionType, ExecutionMode


# Confidence score distribution parameters (mean, std_dev) per decision type.
# Higher-confidence decision types have tighter distributions; fraud is inherently noisier.
_CONFIDENCE_PARAMS: Dict[DecisionType, tuple[float, float]] = {
    DecisionType.PRODUCT_RECOMMENDATION: (0.82, 0.10),
    DecisionType.OFFER_SELECTION:        (0.74, 0.12),
    DecisionType.FRAUD_DETECTION:        (0.68, 0.15),
    DecisionType.PRICING:                (0.79, 0.09),
    DecisionType.NOTIFICATION:           (0.77, 0.11),
    DecisionType.SEARCH_RANKING:         (0.85, 0.08),
}


class DecisionEngine:
    """
    Produces Decision objects for each incoming request. Supports per-type
    confidence simulation and injects anomaly context when the traffic
    generator signals an anomaly period (flash sale, bot traffic, etc.).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def make_decision(
        self,
        decision_type: DecisionType,
        autonomy_boundary: float,
        *,
        is_anomaly_period: bool = False,
        traffic_multiplier: float = 1.0,
        execution_mode: ExecutionMode = ExecutionMode.AUTO,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Decision:
        """
        Construct a Decision with a simulated confidence score.
        The execution_mode is set by the AutonomyController before calling here;
        we accept it as a parameter to keep routing logic out of the engine.
        """
        mean, std = _CONFIDENCE_PARAMS[decision_type]

        # During anomaly periods, confidence degrades (confidence-outcome divergence)
        if is_anomaly_period:
            mean = max(0.40, mean - 0.15)
            std = min(0.25, std + 0.08)

        confidence = float(
            max(0.01, min(0.99, self._rng.gauss(mean, std)))
        )

        context = DecisionContext(
            user_id=user_id or f"u_{self._rng.randint(1, 100_000)}",
            session_id=session_id or f"s_{self._rng.randint(1, 500_000)}",
            request_latency_ms=round(self._rng.uniform(2.0, 12.0), 2),
            confidence_score=confidence,
            is_anomaly_period=is_anomaly_period,
            traffic_multiplier=traffic_multiplier,
        )

        return Decision(
            decision_type=decision_type,
            context=context,
            execution_mode=execution_mode,
            autonomy_boundary_at_decision=autonomy_boundary,
        )
