"""
Production Simulator — bridges the PatternGenerator to the Autonomy Learning Loop.

Responsibilities:
  1. Drive PatternGenerator at each time step
  2. Expose a step(t) interface that returns ProductionEnvState
     (duck-typed compatible with NoiseInjector's AnomalyState for Simulator integration)
  3. Implement map_env_to_signals() — translate environment metrics into
     governance feedback signals (rollback / incident / override / trust_degradation)

Signal mapping rules (configurable via ScenarioConfig thresholds):
  error_rate >= high_error_threshold    → ROLLBACK  (HIGH severity)
  error_rate >= moderate_error_threshold → OVERRIDE (MEDIUM severity)
  latency_ms >= high_latency_threshold  → INCIDENT  (HIGH severity)
  sustained high error + high latency   → TRUST_DEGRADATION (MEDIUM severity)
  confidence > 0.75 AND outcome bad     → CONFIDENCE_DIVERGENCE (caller-injected)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from src.feedback_signals.collector import FeedbackCollector as FC
from src.feedback_signals.models import FeedbackSignal, SignalSeverity, SignalType
from src.simulation.pattern_generator import PatternConfig, PatternGenerator
from src.simulation.scenarios import ScenarioConfig

if TYPE_CHECKING:
    pass


# ─── Environment state ───────────────────────────────────────────────────────

@dataclass
class ProductionEnvState:
    """
    Snapshot of the production environment at a single time step.

    Duck-typed to be compatible with NoiseInjector.AnomalyState so that
    Simulator.run() can accept either type without branching on class identity.

    Fields that mirror AnomalyState (required for Simulator compatibility):
        active           : bool   — whether an anomaly event is in progress
        risk_multiplier  : float  — risk amplification (1.0 = baseline)
        traffic_multiplier: float — volume amplification (1.0 = baseline)
        anomaly_type     : str    — human-readable event label

    Additional production-specific fields:
        t                : int    — current time step
        traffic          : float  — raw traffic volume
        error_rate       : float  — failure fraction [0, 1]
        latency_ms       : float  — response time in ms
        derived_failure_rate: float — P(failure) for outcome simulation
    """
    # AnomalyState-compatible interface
    active: bool
    risk_multiplier: float
    traffic_multiplier: float
    anomaly_type: str

    # Production-specific fields
    t: int
    traffic: float
    error_rate: float
    latency_ms: float
    derived_failure_rate: float   # used by Simulator._simulate_outcome


# ─── Production Simulator ────────────────────────────────────────────────────

class ProductionSimulator:
    """
    Drives realistic production signal generation for the Autonomy Learning Loop.

    Integrates with the existing Simulator via the optional `production_sim`
    parameter — Simulator.run() calls production_sim.step(t) instead of
    noise_injector.step() when a ProductionSimulator is provided.

    Usage::

        scenario = flash_sale_scenario(total_steps=400)
        prod_sim = ProductionSimulator(scenario)
        simulator = Simulator(
            total_steps=400,
            production_sim=prod_sim,
            ...
        )
        results = simulator.run()
    """

    def __init__(self, scenario: ScenarioConfig) -> None:
        self._scenario = scenario
        self._generator = PatternGenerator(scenario.pattern)
        self._base_traffic = scenario.pattern.base_traffic
        self._base_error = scenario.pattern.base_error_rate
        self._replay_log: List[ProductionEnvState] = []

    def step(self, t: int) -> ProductionEnvState:
        """
        Advance one time step and return the production environment state.

        Called by Simulator.run() once per step.
        """
        traffic    = self._generator.traffic(t)
        error_rate = self._generator.error_rate(t)
        latency_ms = self._generator.latency(t)

        # Derived quantities for Simulator compatibility
        traffic_mult = traffic / max(self._base_traffic, 1e-6)

        # risk_multiplier: how much worse than baseline are we?
        # Anchored so that baseline error rate → 1.0
        risk_mult = max(1.0, error_rate / max(self._base_error, 1e-6))

        # is_anomaly: true when error rate is meaningfully above baseline
        # Threshold: 2× baseline error rate or any high-latency condition
        cfg = self._scenario
        is_anomaly = (
            error_rate >= cfg.moderate_error_threshold
            or latency_ms >= cfg.high_latency_threshold_ms
        )

        # Determine event label from active overlays
        anomaly_type = "NORMAL"
        for ev in cfg.pattern.events:
            if ev.is_active(t):
                anomaly_type = ev.event_type
                break

        # P(failure) used by Simulator._simulate_outcome when in production mode
        # Directly derived from measured error_rate rather than a tuned multiplier
        derived_failure_rate = min(0.95, error_rate * (1.0 + math.log1p(latency_ms / 100.0) * 0.1))

        state = ProductionEnvState(
            t=t,
            active=is_anomaly,
            risk_multiplier=round(risk_mult, 3),
            traffic_multiplier=round(traffic_mult, 3),
            anomaly_type=anomaly_type,
            traffic=round(traffic, 2),
            error_rate=round(error_rate, 4),
            latency_ms=round(latency_ms, 2),
            derived_failure_rate=round(derived_failure_rate, 4),
        )
        self._replay_log.append(state)
        return state

    @property
    def scenario(self) -> ScenarioConfig:
        return self._scenario

    def replay_log(self) -> List[ProductionEnvState]:
        """Return recorded history of all environment states (for replay/debug)."""
        return list(self._replay_log)

    def full_series(self) -> List[dict]:
        """Return complete time series as serialisable dicts."""
        return [
            {
                "t": s.t,
                "traffic": s.traffic,
                "error_rate": s.error_rate,
                "latency_ms": s.latency_ms,
                "traffic_multiplier": s.traffic_multiplier,
                "risk_multiplier": s.risk_multiplier,
                "is_anomaly": s.active,
                "anomaly_type": s.anomaly_type,
                "derived_failure_rate": s.derived_failure_rate,
            }
            for s in self._replay_log
        ]


# ─── Signal mapping ───────────────────────────────────────────────────────────

def map_env_to_signals(
    env: ProductionEnvState,
    scenario: ScenarioConfig,
    decision_type: str = "UNKNOWN",
    decision_id: str = "env",
) -> List[FeedbackSignal]:
    """
    Convert an environment state snapshot into governance feedback signals.

    This is the bridge between measured production health and the autonomy
    learning loop. Signals emitted here feed the FeedbackAggregator and drive
    boundary updates.

    Mapping rules:
        error_rate >= high_error_threshold    → ROLLBACK  (HIGH)
        error_rate >= moderate_error_threshold → OVERRIDE (MEDIUM)
        latency_ms >= high_latency_threshold  → INCIDENT  (HIGH)
        both error AND latency high           → TRUST_DEGRADATION (MEDIUM)
        CRITICAL zone (error_rate >= 0.7)     → upgrade ROLLBACK to CRITICAL

    Args:
        env:           current production environment state
        scenario:      scenario config for threshold values
        decision_type: decision class label for signal attribution
        decision_id:   decision id for audit correlation

    Returns:
        List of FeedbackSignal instances (may be empty for healthy state)
    """
    signals: List[FeedbackSignal] = []
    cfg = scenario

    high_error   = cfg.high_error_threshold
    mod_error    = cfg.moderate_error_threshold
    high_latency = cfg.high_latency_threshold_ms

    er  = env.error_rate
    lat = env.latency_ms

    # ── ROLLBACK: high error rate ─────────────────────────────────────────
    if er >= high_error:
        severity = SignalSeverity.CRITICAL if er >= 0.70 else SignalSeverity.HIGH
        signals.append(FC.make_rollback(
            decision_id=decision_id,
            decision_type=decision_type,
            recovery_cost_seconds=lat * 0.5,   # higher latency = more recovery cost
            severity=severity,
        ))

    # ── OVERRIDE: moderate error rate (below rollback threshold) ─────────
    elif er >= mod_error:
        signals.append(FC.make_override(
            decision_id=decision_id,
            decision_type=decision_type,
            severity=SignalSeverity.MEDIUM,
        ))

    # ── INCIDENT: high latency ────────────────────────────────────────────
    if lat >= high_latency:
        severity = SignalSeverity.CRITICAL if lat >= high_latency * 2 else SignalSeverity.HIGH
        signals.append(FC.make_incident(
            decision_id=decision_id,
            decision_type=decision_type,
            severity=severity,
        ))

    # ── TRUST_DEGRADATION: both error and latency elevated ────────────────
    if er >= mod_error and lat >= high_latency * 0.7:
        signals.append(FC.make_trust_degradation(
            decision_type=decision_type,
            severity=SignalSeverity.MEDIUM,
        ))

    return signals
