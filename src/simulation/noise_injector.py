"""
Noise Injector — injects anomaly periods, flash sales, and operational incidents
into the simulation to stress-test the Autonomy Learning Loop's contraction response.

Anomaly periods model:
  - Black Friday / Cyber Monday flash sales (high traffic + elevated risk)
  - Bot traffic spikes (anomalous confidence patterns)
  - Upstream service degradation (increased failure rates)
  - Seasonal trust degradation signals
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnomalyState:
    active: bool = False
    steps_remaining: int = 0
    risk_multiplier: float = 1.0
    traffic_multiplier: float = 1.0
    anomaly_type: str = "NONE"


class NoiseInjector:
    """
    Manages anomaly injection lifecycle for the simulation.

    At each step, the injector determines:
    1. Whether an anomaly starts (probability-based)
    2. Current risk and traffic multipliers
    3. Whether the anomaly period ends
    """

    def __init__(
        self,
        anomaly_probability: float = 0.05,
        anomaly_duration_steps: int = 10,
        peak_risk_multiplier: float = 3.5,
        peak_traffic_multiplier: float = 2.5,
        seed: Optional[int] = None,
    ) -> None:
        self._anomaly_prob = anomaly_probability
        self._anomaly_duration = anomaly_duration_steps
        self._peak_risk_mult = peak_risk_multiplier
        self._peak_traffic_mult = peak_traffic_multiplier
        self._rng = random.Random(seed)
        self._state = AnomalyState()

        self._anomaly_types = [
            ("FLASH_SALE",     peak_risk_multiplier,       peak_traffic_multiplier),
            ("BOT_SPIKE",      peak_risk_multiplier * 0.8, peak_traffic_multiplier * 1.5),
            ("SERVICE_DEGRAD", peak_risk_multiplier * 1.2, 1.0),
            ("TRUST_EROSION",  peak_risk_multiplier * 0.5, 1.2),
        ]

    def step(self) -> AnomalyState:
        """
        Advance the anomaly lifecycle by one step.
        Returns the current AnomalyState for use by the Simulator.
        """
        if self._state.active:
            self._state.steps_remaining -= 1
            if self._state.steps_remaining <= 0:
                self._state = AnomalyState(active=False)
        else:
            if self._rng.random() < self._anomaly_prob:
                atype, risk_mult, traffic_mult = self._rng.choice(self._anomaly_types)
                self._state = AnomalyState(
                    active=True,
                    steps_remaining=self._anomaly_duration,
                    risk_multiplier=risk_mult,
                    traffic_multiplier=traffic_mult,
                    anomaly_type=atype,
                )

        return AnomalyState(
            active=self._state.active,
            steps_remaining=self._state.steps_remaining,
            risk_multiplier=self._state.risk_multiplier if self._state.active else 1.0,
            traffic_multiplier=self._state.traffic_multiplier if self._state.active else 1.0,
            anomaly_type=self._state.anomaly_type if self._state.active else "NONE",
        )

    def force_anomaly(
        self,
        duration_steps: int = 5,
        risk_multiplier: float = 3.0,
        anomaly_type: str = "FORCED",
    ) -> None:
        """Programmatically trigger an anomaly (useful for experiment injection)."""
        self._state = AnomalyState(
            active=True,
            steps_remaining=duration_steps,
            risk_multiplier=risk_multiplier,
            traffic_multiplier=1.5,
            anomaly_type=anomaly_type,
        )

    @property
    def is_anomaly_active(self) -> bool:
        return self._state.active
