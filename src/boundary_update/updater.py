"""
Boundary Updater — implements the formal boundary adaptation logic from §5 of the paper.

Core formulas:
    Expansion (safe):    ΔB_t = α(1 − B_t)   if S_t < τ_safe
    Contraction (risk):  ΔB_t = −β·B_t         if S_t ≥ τ_risk
    Neutral zone:        ΔB_t = 0               if τ_safe ≤ S_t < τ_risk

Constraints:
    β >> α  (fast contraction, slow expansion — asymmetric safety guarantee)
    Rate limiting: min_update_interval prevents oscillatory churn
    Frozen classes skip expansion updates
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from src.feedback_aggregator.aggregator import WindowMetrics
from src.boundary_update.spike_detector import SpikeDetector, SpikeCheckResult

logger = logging.getLogger(__name__)


class UpdateDirection(str, Enum):
    EXPAND          = "EXPAND"
    CONTRACT        = "CONTRACT"
    NEUTRAL         = "NEUTRAL"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"   # immediate drop to conservative_boundary
    TREND_CONTRACT  = "TREND_CONTRACT"    # soft contraction on slow upward trend


@dataclass
class BoundaryUpdateResult:
    """Record of a single boundary update event — emitted to audit logger."""
    decision_type: str
    window_index: int
    boundary_before: float
    boundary_after: float
    delta: float
    direction: UpdateDirection
    risk_score: float           # S_t that triggered this update
    safe_threshold: float
    risk_threshold: float
    rate_limited: bool = False           # True if update was suppressed by rate limiter
    frozen_skip: bool = False            # True if expansion skipped due to freeze
    headroom_tapered: bool = False       # True if expansion was slowed by headroom buffer
    headroom_taper_factor: float = 1.0   # 1.0 = full rate; <1 = tapered
    circuit_breaker_triggered: bool = False   # True if spike detector fired
    circuit_breaker_active: bool = False      # True if in recovery-lock period
    spike_growth_rate: float = 1.0            # S_t growth rate that triggered check
    trend_detected: bool = False              # True if slow upward trend detected
    trend_slope: float = 0.0                  # linear regression slope of S_t
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def boundary_change_pct(self) -> float:
        return round((self.delta / max(self.boundary_before, 1e-6)) * 100, 2)

    def to_dict(self) -> dict:
        return {
            "decision_type": self.decision_type,
            "window_index": self.window_index,
            "boundary_before": round(self.boundary_before, 4),
            "boundary_after": round(self.boundary_after, 4),
            "delta": round(self.delta, 4),
            "direction": self.direction.value,
            "risk_score_S_t": round(self.risk_score, 4),
            "safe_threshold": self.safe_threshold,
            "risk_threshold": self.risk_threshold,
            "rate_limited": self.rate_limited,
            "frozen_skip": self.frozen_skip,
            "headroom_tapered": self.headroom_tapered,
            "headroom_taper_factor": round(self.headroom_taper_factor, 4),
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "circuit_breaker_active": self.circuit_breaker_active,
            "spike_growth_rate": round(self.spike_growth_rate, 3),
            "trend_detected": self.trend_detected,
            "trend_slope": round(self.trend_slope, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class BoundaryUpdater:
    """
    Applies the asymmetric boundary update rule from §5.

    Designed to be called once per window evaluation by the Simulator.
    Updates are rate-limited per decision class to prevent oscillation.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.30,
        safe_threshold: float = 0.25,
        risk_threshold: float = 0.60,
        min_boundary: float = 0.05,
        max_boundary: float = 0.95,
        min_update_interval_seconds: float = 60.0,
        headroom_buffer: float = 0.15,
        spike_growth_threshold: float = 1.5,
        spike_consecutive_windows: int = 2,
        spike_conservative_boundary: float = 0.15,
        spike_recovery_lock_windows: int = 10,
        sustained_safe_windows: int = 3,
        spike_trend_window_count: int = 4,
        spike_trend_slope_threshold: float = 0.02,
    ) -> None:
        assert beta > alpha, "β must be greater than α for asymmetric safety guarantee"
        assert safe_threshold < risk_threshold, "safe_threshold must be < risk_threshold"
        assert 0.0 <= headroom_buffer < (max_boundary - min_boundary), (
            "headroom_buffer must be in [0, max_boundary - min_boundary)"
        )

        self.alpha = alpha
        self.beta = beta
        self.safe_threshold = safe_threshold
        self.risk_threshold = risk_threshold
        self.min_boundary = min_boundary
        self.max_boundary = max_boundary
        self.min_update_interval = min_update_interval_seconds
        self.headroom_buffer = headroom_buffer
        # Soft ceiling: expansion tapers linearly above this point
        self._soft_ceiling = max_boundary - headroom_buffer

        # Spike circuit breaker + trend detector
        self._spike_detector = SpikeDetector(
            growth_threshold=spike_growth_threshold,
            consecutive_windows=spike_consecutive_windows,
            conservative_boundary=spike_conservative_boundary,
            recovery_lock_windows=spike_recovery_lock_windows,
            sustained_safe_windows=sustained_safe_windows,
            safe_threshold=safe_threshold,
            trend_window_count=spike_trend_window_count,
            trend_slope_threshold=spike_trend_slope_threshold,
        )

        # Per-class rate-limiter: tracks last update time
        self._last_update_time: Dict[str, float] = {}
        self._lock = threading.Lock()

    def compute_update(
        self,
        current_boundary: float,
        metrics: WindowMetrics,
        *,
        is_frozen: bool = False,
    ) -> BoundaryUpdateResult:
        """
        Compute the new boundary value given current B and window metrics.

        Does NOT apply the update to the controller — caller is responsible
        for calling controller.set_boundary() after receiving the result.
        This keeps the updater stateless with respect to the controller.
        """
        s_t = metrics.normalized_risk_score
        b = current_boundary
        dt = metrics.decision_type

        # ── Spike circuit breaker (checked before anything else) ──────────
        # If S_t is growing consistently at >= growth_threshold per window,
        # jump immediately to conservative_boundary instead of gradual contraction.
        spike = self._spike_detector.check(dt, s_t)
        if spike.triggered:
            new_b = max(self.min_boundary, self._spike_detector.conservative_boundary)
            actual_delta = new_b - b
            with self._lock:
                self._last_update_time[dt] = time.monotonic()
            logger.warning(
                "CIRCUIT BREAKER fired for %s: B %.3f → %.3f "
                "(S_t=%.3f grew %.1f× over %d windows)",
                dt, b, new_b, s_t, spike.growth_rate, spike.consecutive_growth_count + 1,
            )
            return BoundaryUpdateResult(
                decision_type=dt,
                window_index=metrics.window_index,
                boundary_before=b,
                boundary_after=new_b,
                delta=actual_delta,
                direction=UpdateDirection.CIRCUIT_BREAKER,
                risk_score=s_t,
                safe_threshold=self.safe_threshold,
                risk_threshold=self.risk_threshold,
                circuit_breaker_triggered=True,
                circuit_breaker_active=True,
                spike_growth_rate=spike.growth_rate,
            )

        # Determine update direction based on S_t vs thresholds
        if s_t < self.safe_threshold:
            direction = UpdateDirection.EXPAND
        elif s_t >= self.risk_threshold:
            direction = UpdateDirection.CONTRACT
        else:
            direction = UpdateDirection.NEUTRAL

        # Block expansion during circuit breaker recovery lock
        if direction == UpdateDirection.EXPAND and spike.circuit_breaker_active:
            direction = UpdateDirection.NEUTRAL

        # Trend contraction: slow upward trend not yet triggering τ_risk
        # Apply a soft contraction (35% of β) to get ahead of slow degradation
        if spike.trend_detected and direction in (UpdateDirection.EXPAND, UpdateDirection.NEUTRAL):
            direction = UpdateDirection.TREND_CONTRACT

        # Check rate limiter
        now = time.monotonic()
        with self._lock:
            last = self._last_update_time.get(dt, 0.0)
            elapsed = now - last
            rate_limited = elapsed < self.min_update_interval and direction != UpdateDirection.CONTRACT

        # Contractions bypass rate limit — fast safety response
        if rate_limited and direction != UpdateDirection.CONTRACT:
            return BoundaryUpdateResult(
                decision_type=dt,
                window_index=metrics.window_index,
                boundary_before=b,
                boundary_after=b,
                delta=0.0,
                direction=direction,
                risk_score=s_t,
                safe_threshold=self.safe_threshold,
                risk_threshold=self.risk_threshold,
                rate_limited=True,
            )

        # Expansion skipped if class is frozen
        if direction == UpdateDirection.EXPAND and is_frozen:
            return BoundaryUpdateResult(
                decision_type=dt,
                window_index=metrics.window_index,
                boundary_before=b,
                boundary_after=b,
                delta=0.0,
                direction=direction,
                risk_score=s_t,
                safe_threshold=self.safe_threshold,
                risk_threshold=self.risk_threshold,
                frozen_skip=True,
            )

        # Apply the boundary update formula
        headroom_tapered = False
        taper_factor = 1.0

        if direction == UpdateDirection.EXPAND:
            delta = self.alpha * (1.0 - b)                # slow expansion → full autonomy
            # Headroom taper: as B enters the reserved zone above soft_ceiling,
            # scale expansion down linearly to zero at max_boundary.
            # This keeps a safety buffer available for unanticipated contractions.
            if self.headroom_buffer > 0 and b > self._soft_ceiling:
                taper_factor = (self.max_boundary - b) / self.headroom_buffer
                taper_factor = max(0.0, min(1.0, taper_factor))
                delta *= taper_factor
                headroom_tapered = True
        elif direction == UpdateDirection.CONTRACT:
            delta = -self.beta * b                         # fast contraction → zero (never tapered)
        elif direction == UpdateDirection.TREND_CONTRACT:
            delta = -self.beta * b * 0.35                  # soft: 35% of normal contraction
        else:
            delta = 0.0

        new_b = max(self.min_boundary, min(self.max_boundary, b + delta))
        actual_delta = new_b - b

        # Update rate-limiter timestamp
        with self._lock:
            self._last_update_time[dt] = now

        result = BoundaryUpdateResult(
            decision_type=dt,
            window_index=metrics.window_index,
            boundary_before=b,
            boundary_after=new_b,
            delta=actual_delta,
            direction=direction,
            risk_score=s_t,
            safe_threshold=self.safe_threshold,
            risk_threshold=self.risk_threshold,
            headroom_tapered=headroom_tapered,
            headroom_taper_factor=taper_factor,
            circuit_breaker_active=spike.circuit_breaker_active,
            spike_growth_rate=spike.growth_rate,
            trend_detected=spike.trend_detected,
            trend_slope=spike.trend_slope,
        )

        if spike.trend_detected:
            logger.debug(
                "TREND CONTRACTION for %s: B %.3f → %.3f (slope=%.4f)",
                dt, b, new_b, spike.trend_slope,
            )

        logger.debug(
            "Boundary update: %s B: %.3f → %.3f (ΔB=%.4f, S_t=%.3f, %s)",
            dt, b, new_b, actual_delta, s_t, direction.value,
        )
        return result
