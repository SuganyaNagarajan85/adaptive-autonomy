"""
Pattern Generator — pure time-series math for realistic production traffic patterns.

Produces three continuous signals over simulation time:
  - traffic(t)    : request volume (normalized, base=1.0)
  - error_rate(t) : failure fraction  ∈ [0, 1]
  - latency(t)    : response time in ms

Signal composition (additive):
  value(t) = base + diurnal(t) + trend(t) + noise(t) + Σ event_spikes(t)

Diurnal component: sin-wave representing intraday traffic patterns.
  - traffic peaks at midday, troughs overnight
  - error_rate inversely correlated (more errors at peak load)

Trend component: linear drift — used for degradation and recovery scenarios.

Noise component: Gaussian noise clipped to prevent negative values.

Event spikes: step-function overlays injected by scenarios for burst events.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EventOverlay:
    """
    A single timed event that modifies signals during a time window.
    All multipliers are applied on top of the base computed value.
    """
    start_step: int
    duration_steps: int
    traffic_multiplier: float = 1.0
    error_multiplier: float = 1.0
    latency_multiplier: float = 1.0
    event_type: str = "GENERIC"

    def is_active(self, t: int) -> bool:
        return self.start_step <= t < self.start_step + self.duration_steps

    def time_fraction(self, t: int) -> float:
        """0.0 at start, 1.0 at end — used for gradual ramp-in/out."""
        if not self.is_active(t):
            return 0.0
        return (t - self.start_step) / max(self.duration_steps, 1)


@dataclass
class PatternConfig:
    """
    Configuration for a PatternGenerator instance.
    All values define the baseline before diurnal/noise/events are applied.
    """
    # Baseline signal levels
    base_traffic: float = 100.0          # requests per step
    base_error_rate: float = 0.05        # 5% errors at baseline
    base_latency_ms: float = 50.0        # 50ms at baseline

    # Diurnal variation amplitude (fraction of base)
    traffic_diurnal_amplitude: float = 0.30   # ±30% intraday swing
    error_diurnal_amplitude: float = 0.10     # ±10% error swing
    latency_diurnal_amplitude: float = 0.20   # ±20% latency swing
    diurnal_period_steps: int = 96            # one "day" = 96 steps

    # Noise (Gaussian std dev as fraction of base)
    traffic_noise_sigma: float = 0.05
    error_noise_sigma: float = 0.02
    latency_noise_sigma: float = 0.03

    # Linear trend (delta per step as fraction of base; positive = increasing)
    traffic_trend_per_step: float = 0.0
    error_trend_per_step: float = 0.0
    latency_trend_per_step: float = 0.0

    # Events (flash sales, degradations, etc.)
    events: List[EventOverlay] = field(default_factory=list)

    seed: int = 42


class PatternGenerator:
    """
    Generates realistic production-grade time-series signals for simulation.

    Usage::

        config = PatternConfig(base_error_rate=0.05, events=[...])
        gen = PatternGenerator(config)
        for t in range(500):
            traffic  = gen.traffic(t)
            error    = gen.error_rate(t)
            latency  = gen.latency(t)
    """

    def __init__(self, config: PatternConfig) -> None:
        self._cfg = config
        self._rng = random.Random(config.seed)
        # Pre-generate noise sequences so signals are reproducible and
        # deterministic regardless of call order.
        self._max_steps = 10_000
        self._traffic_noise  = [self._rng.gauss(0, config.traffic_noise_sigma)
                                 for _ in range(self._max_steps)]
        self._error_noise    = [self._rng.gauss(0, config.error_noise_sigma)
                                 for _ in range(self._max_steps)]
        self._latency_noise  = [self._rng.gauss(0, config.latency_noise_sigma)
                                 for _ in range(self._max_steps)]

    # ------------------------------------------------------------------
    # Public signal API
    # ------------------------------------------------------------------

    def traffic(self, t: int) -> float:
        """
        Returns request volume at time step t (raw units, not normalized).
        Min-clipped at 1 request.
        """
        base = self._cfg.base_traffic
        val = (
            base
            + self._diurnal(t, self._cfg.traffic_diurnal_amplitude, base, phase_offset=0.0)
            + self._trend(t, self._cfg.traffic_trend_per_step, base)
            + self._noise(t, "traffic", base)
        )
        # Apply active event multipliers
        for ev in self._cfg.events:
            if ev.is_active(t):
                val *= ev.traffic_multiplier
        return max(1.0, val)

    def error_rate(self, t: int) -> float:
        """
        Returns error rate at time step t, clamped to [0, 1].
        Error rate is positively correlated with traffic load (diurnal phase-shifted).
        """
        base = self._cfg.base_error_rate
        val = (
            base
            + self._diurnal(t, self._cfg.error_diurnal_amplitude, base, phase_offset=0.0)
            + self._trend(t, self._cfg.error_trend_per_step, base)
            + self._noise(t, "error", base)
        )
        # Apply active event multipliers (can push error_rate very high)
        for ev in self._cfg.events:
            if ev.is_active(t):
                val *= ev.error_multiplier
        return max(0.0, min(1.0, val))

    def latency(self, t: int) -> float:
        """
        Returns latency in ms at time step t.
        Latency correlates with error_rate (high errors = saturated backends = slow).
        Min-clipped at 1ms.
        """
        base = self._cfg.base_latency_ms
        # Latency shares diurnal with traffic; add error-driven component
        error_driven = self.error_rate(t) * base * 2.0   # saturated backend factor
        val = (
            base
            + self._diurnal(t, self._cfg.latency_diurnal_amplitude, base, phase_offset=0.1)
            + self._trend(t, self._cfg.latency_trend_per_step, base)
            + self._noise(t, "latency", base)
            + error_driven * 0.5
        )
        for ev in self._cfg.events:
            if ev.is_active(t):
                val *= ev.latency_multiplier
        return max(1.0, val)

    def snapshot(self, t: int) -> dict:
        """Return all three signals at time t as a dict."""
        return {
            "t": t,
            "traffic": round(self.traffic(t), 2),
            "error_rate": round(self.error_rate(t), 4),
            "latency_ms": round(self.latency(t), 2),
        }

    def series(self, total_steps: int) -> List[dict]:
        """Generate the full time series (useful for replay and plotting)."""
        return [self.snapshot(t) for t in range(total_steps)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _diurnal(self, t: int, amplitude: float, base: float, phase_offset: float) -> float:
        """
        Sinusoidal diurnal component.
        amplitude is expressed as fraction of base (e.g. 0.30 = ±30% swing).
        phase_offset in radians shifts the peak (used for latency vs traffic lag).
        """
        period = self._cfg.diurnal_period_steps
        angle = (2.0 * math.pi * t / period) + phase_offset
        return amplitude * base * math.sin(angle)

    def _trend(self, t: int, delta_per_step: float, base: float) -> float:
        """Linear trend: delta_per_step × base × t."""
        return delta_per_step * base * t

    def _noise(self, t: int, channel: str, base: float) -> float:
        """Pre-generated Gaussian noise scaled to base value."""
        idx = t % self._max_steps
        if channel == "traffic":
            return self._traffic_noise[idx] * base
        if channel == "error":
            return self._error_noise[idx] * base
        return self._latency_noise[idx] * base
