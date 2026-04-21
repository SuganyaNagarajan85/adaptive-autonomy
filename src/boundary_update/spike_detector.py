"""
Spike Detector — circuit breaker + trend detector for risk signal patterns.

Two detection modes:

  1. CIRCUIT BREAKER (fast spike)
     S_t grows at >= growth_threshold per window for consecutive_windows in a row.
     Action: immediately set B = conservative_boundary, bypass gradual contraction.
     Recovery: expansion locked until BOTH time (recovery_lock_windows) AND
     sustained safety (sustained_safe_windows consecutive safe windows) are met.
     Prevents the CB → expand → CB oscillation seen in degraded recovery scenarios.

  2. TREND CONTRACTION (slow degradation)
     S_t shows a consistent upward slope over the last trend_window_count windows,
     even if each individual window is below τ_risk.
     Computed via linear regression slope over S_t history.
     Action: force a soft contraction (scaled β) instead of waiting for τ_risk.
     This catches the "frog in boiling water" pattern where S_t climbs slowly
     enough that single-window evaluation never triggers the standard rule.

State machine per decision class:
    NORMAL ──(CB trigger)──► TRIGGERED ──► RECOVERY_LOCKED ──(time+safety)──► NORMAL
    NORMAL ──(trend)──► soft contraction each window until trend clears

Parameters:
    growth_threshold         : S_t[t] / S_t[t-1] must exceed this             (default 1.5)
    consecutive_windows      : consecutive growth windows to trigger CB         (default 2)
    conservative_boundary    : B set to this on CB trigger                     (default 0.15)
    recovery_lock_windows    : minimum windows locked post-CB                  (default 10)
    sustained_safe_windows   : consecutive safe windows needed to exit lock     (default 3)
    safe_threshold           : S_t below this counts as a "safe" window        (default 0.25)
    trend_window_count       : S_t history length for slope computation        (default 4)
    trend_slope_threshold    : minimum S_t slope/window to flag as trend       (default 0.02)
"""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SpikeCheckResult:
    """Result of a per-window spike/trend check for one decision class."""
    decision_type: str
    # Circuit breaker
    triggered: bool               # True if CB just fired this window
    circuit_breaker_active: bool  # True if in recovery-lock period
    growth_rate: float            # S_t[t] / S_t[t-1]
    consecutive_growth_count: int
    recovery_lock_remaining: int
    consecutive_safe_count: int   # safe windows accumulated since last CB
    # Trend detection
    trend_detected: bool          # True if slow upward trend detected
    trend_slope: float            # computed S_t slope (per window)
    # Raw values
    s_t: float
    previous_s_t: float


class SpikeDetector:
    """
    Per-class S_t trend detector and circuit breaker.

    Called by BoundaryUpdater.compute_update() before applying the standard
    expansion/contraction formula.
    """

    def __init__(
        self,
        # Circuit breaker params
        growth_threshold: float = 1.5,
        consecutive_windows: int = 2,
        conservative_boundary: float = 0.15,
        recovery_lock_windows: int = 10,
        # Sustained-safety params (post-CB)
        sustained_safe_windows: int = 3,
        safe_threshold: float = 0.25,
        # Trend detection params
        trend_window_count: int = 4,
        trend_slope_threshold: float = 0.02,
    ) -> None:
        assert growth_threshold > 1.0
        assert consecutive_windows >= 1
        assert 0.0 < conservative_boundary < 1.0
        assert recovery_lock_windows >= 0
        assert sustained_safe_windows >= 1
        assert trend_window_count >= 3
        assert trend_slope_threshold > 0

        self.growth_threshold       = growth_threshold
        self.consecutive_windows    = consecutive_windows
        self.conservative_boundary  = conservative_boundary
        self.recovery_lock_windows  = recovery_lock_windows
        self.sustained_safe_windows = sustained_safe_windows
        self.safe_threshold         = safe_threshold
        self.trend_window_count     = trend_window_count
        self.trend_slope_threshold  = trend_slope_threshold

        # Per-class state — all defaultdict so new classes are zero-initialised
        history_len = max(trend_window_count + 1, 5)
        self._s_t_history:      Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_len)
        )
        self._growth_count:     Dict[str, int]  = defaultdict(int)
        self._cb_active:        Dict[str, bool] = defaultdict(bool)
        self._recovery_lock:    Dict[str, int]  = defaultdict(int)
        self._consecutive_safe: Dict[str, int]  = defaultdict(int)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, decision_type: str, s_t: float) -> SpikeCheckResult:
        """
        Evaluate the current window's S_t for the given decision class.

        Must be called exactly once per window boundary, in the same call order
        as BoundaryUpdater.compute_update().
        """
        dt = decision_type
        history = self._s_t_history[dt]
        prev_s_t = history[-1] if history else 0.0
        history.append(s_t)

        # ── 1. Consecutive safe window tracking ───────────────────────────
        if s_t < self.safe_threshold:
            self._consecutive_safe[dt] += 1
        else:
            self._consecutive_safe[dt] = 0

        # ── 2. Circuit breaker: growth rate ──────────────────────────────
        growth_rate = (s_t / prev_s_t) if prev_s_t > 0.01 else 1.0

        if growth_rate >= self.growth_threshold and s_t > 0.05:
            self._growth_count[dt] += 1
        else:
            self._growth_count[dt] = 0

        # ── 3. Circuit breaker: should it fire? ───────────────────────────
        just_triggered = False
        if (
            not self._cb_active[dt]
            and self._growth_count[dt] >= self.consecutive_windows
        ):
            self._cb_active[dt]     = True
            self._recovery_lock[dt] = self.recovery_lock_windows
            self._growth_count[dt]  = 0
            self._consecutive_safe[dt] = 0   # reset safety clock on new CB
            just_triggered = True

        # ── 4. Recovery lock: time countdown + sustained-safety gate ─────
        elif self._cb_active[dt]:
            if self._recovery_lock[dt] > 0:
                self._recovery_lock[dt] -= 1
            else:
                # Time lock expired — also require sustained safe windows
                if self._consecutive_safe[dt] >= self.sustained_safe_windows:
                    self._cb_active[dt] = False
                    self._consecutive_safe[dt] = 0
                # else: stay locked until safety condition met

        # ── 5. Trend detection (slow degradation) ────────────────────────
        trend_detected, trend_slope = self._detect_trend(dt)

        return SpikeCheckResult(
            decision_type=dt,
            triggered=just_triggered,
            circuit_breaker_active=self._cb_active[dt],
            growth_rate=round(growth_rate, 3),
            consecutive_growth_count=self._growth_count[dt],
            recovery_lock_remaining=self._recovery_lock[dt],
            consecutive_safe_count=self._consecutive_safe[dt],
            trend_detected=trend_detected,
            trend_slope=round(trend_slope, 4),
            s_t=round(s_t, 4),
            previous_s_t=round(prev_s_t, 4),
        )

    def is_recovery_locked(self, decision_type: str) -> bool:
        return self._cb_active[decision_type]

    def reset(self, decision_type: str) -> None:
        self._growth_count[decision_type]     = 0
        self._cb_active[decision_type]        = False
        self._recovery_lock[decision_type]    = 0
        self._consecutive_safe[decision_type] = 0
        self._s_t_history[decision_type].clear()

    # ── Private ───────────────────────────────────────────────────────────────

    def _detect_trend(self, dt: str) -> tuple[bool, float]:
        """
        Compute the linear slope of S_t over the last trend_window_count windows.
        Returns (trend_detected, slope).

        Trend is flagged when:
          - At least trend_window_count history points exist
          - Slope > trend_slope_threshold (S_t consistently rising)
          - The latest S_t is above safe_threshold (not in a safe zone)
          - Circuit breaker is not already active (no double-trigger)
        """
        history = list(self._s_t_history[dt])
        if len(history) < self.trend_window_count:
            return False, 0.0

        # Use the last trend_window_count values
        values: List[float] = history[-self.trend_window_count:]
        n = len(values)

        # Linear regression slope: Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator   = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0

        current_s_t = values[-1]
        trend_detected = (
            slope >= self.trend_slope_threshold
            and current_s_t > self.safe_threshold   # only flag outside safe zone
            and not self._cb_active[dt]             # CB already handles spikes
        )

        return trend_detected, slope
