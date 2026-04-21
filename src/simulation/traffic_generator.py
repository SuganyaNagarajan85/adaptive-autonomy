"""
Traffic Generator — synthesizes realistic e-commerce decision request traffic.

Models:
  - Poisson-distributed request arrivals per step
  - Weighted decision type distribution (recommendations dominate; fraud checks are rare)
  - User/session diversity for attribution analysis
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from src.decision_engine.models import DecisionType


# Relative traffic weight per decision type reflecting typical e-commerce workload
_TRAFFIC_WEIGHTS: Dict[DecisionType, float] = {
    DecisionType.PRODUCT_RECOMMENDATION: 0.35,
    DecisionType.SEARCH_RANKING:         0.25,
    DecisionType.NOTIFICATION:           0.15,
    DecisionType.PRICING:                0.12,
    DecisionType.OFFER_SELECTION:        0.08,
    DecisionType.FRAUD_DETECTION:        0.05,
}

_DECISION_TYPES = list(_TRAFFIC_WEIGHTS.keys())
_WEIGHTS = [_TRAFFIC_WEIGHTS[dt] for dt in _DECISION_TYPES]


class TrafficGenerator:
    """
    Generates a batch of (DecisionType, user_id) tuples for each simulation step.
    Volume follows a Poisson distribution around the configured mean.
    """

    def __init__(
        self,
        decisions_per_step: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self._base_volume = decisions_per_step
        self._rng = random.Random(seed)

    def generate_step(
        self,
        traffic_multiplier: float = 1.0,
    ) -> List[Tuple[DecisionType, str, str]]:
        """
        Generate one step's worth of (decision_type, user_id, session_id) tuples.
        Volume scales with traffic_multiplier (flash sale = high multiplier).
        """
        mean_volume = max(1, int(self._base_volume * traffic_multiplier))
        # Poisson noise around mean volume — realistic for web traffic
        volume = self._rng.choices(
            range(max(1, mean_volume - 5), mean_volume + 6),
            k=1,
        )[0]

        return [
            (
                self._rng.choices(_DECISION_TYPES, weights=_WEIGHTS, k=1)[0],
                f"u_{self._rng.randint(1, 50_000)}",
                f"s_{self._rng.randint(1, 200_000)}",
            )
            for _ in range(volume)
        ]

    def set_base_volume(self, volume: int) -> None:
        self._base_volume = volume
