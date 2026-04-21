"""
Scenarios — predefined, reusable, configurable production scenario configs.

Each scenario describes a realistic operational condition that a production
ecommerce platform might encounter. Scenarios are composed of:
  - A PatternConfig driving the pattern generator
  - Metadata for labeling outputs and audit logs
  - Factory functions for easy parameterisation

The three canonical scenarios from the specification:

  1. FLASH_SALE     — sudden traffic spike, high correlated error rate, short burst
  2. DEGRADATION    — stable traffic, gradually increasing errors + latency, no spikes
  3. RECOVERY       — starts degraded, errors fall, system stabilises

All scenario factories accept keyword overrides so callers can tune any
individual field without constructing the full dataclass manually.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.simulation.pattern_generator import EventOverlay, PatternConfig


# ─── Scenario metadata ───────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """
    Full scenario specification consumed by ProductionSimulator.

    Combines:
      - PatternConfig  (signal generation parameters)
      - Scenario-level metadata (name, steps, description)
      - Governance parameters forwarded to Simulator
    """
    # Identity
    name: str
    description: str

    # Simulation length
    total_steps: int = 400

    # Signal generation
    pattern: PatternConfig = field(default_factory=PatternConfig)

    # Governance parameters (forwarded to Simulator / BoundaryUpdater)
    initial_boundary: float = 0.5
    alpha: float = 0.05
    beta: float = 0.30
    safe_threshold: float = 0.25
    risk_threshold: float = 0.60
    window_size: int = 50
    decisions_per_step: int = 20
    headroom_buffer: float = 0.15   # reserved gap below max_boundary; tapers expansion

    # Reproducibility
    seed: int = 42

    # Signal mapping thresholds — used by map_env_to_signals()
    high_error_threshold: float = 0.35     # → ROLLBACK signal
    moderate_error_threshold: float = 0.15  # → OVERRIDE signal
    high_latency_threshold_ms: float = 150.0  # → INCIDENT signal

    def output_dir(self, base: str = "outputs") -> str:
        """Canonical output directory for this scenario's plots and JSON."""
        return f"{base}/{self.name.lower().replace(' ', '_')}"


# ─── Scenario 1: Flash Sale ───────────────────────────────────────────────────

def flash_sale_scenario(
    total_steps: int = 400,
    spike_start: int = 100,
    spike_duration: int = 50,
    traffic_multiplier: float = 5.0,
    error_multiplier: float = 4.0,
    latency_multiplier: float = 3.0,
    base_error_rate: float = 0.05,
    seed: int = 42,
    **overrides,
) -> ScenarioConfig:
    """
    Flash Sale scenario.

    Timeline:
      Steps 0–spike_start      : stable baseline (normal ecommerce load)
      Steps spike_start–+dur   : burst — 5× traffic, 4× errors, 3× latency
      Steps after burst        : rapid normalisation (errors decay naturally)

    Design intent: tests whether the boundary controller contracts fast enough
    during the burst and re-expands correctly once traffic normalises.

    Args:
        total_steps: total simulation length
        spike_start: step at which flash sale begins
        spike_duration: how many steps the burst lasts
        traffic_multiplier: peak traffic as multiple of base
        error_multiplier: peak error rate as multiple of base
        latency_multiplier: peak latency as multiple of base
        base_error_rate: normal-operation error rate
        seed: RNG seed
        **overrides: any ScenarioConfig field can be overridden
    """
    # Post-burst recovery: errors halve within 20 steps after the spike
    recovery_start = spike_start + spike_duration
    recovery_duration = min(30, total_steps - recovery_start)

    events: List[EventOverlay] = [
        EventOverlay(
            start_step=spike_start,
            duration_steps=spike_duration,
            traffic_multiplier=traffic_multiplier,
            error_multiplier=error_multiplier,
            latency_multiplier=latency_multiplier,
            event_type="FLASH_SALE_BURST",
        ),
    ]
    if recovery_start < total_steps and recovery_duration > 0:
        events.append(
            EventOverlay(
                start_step=recovery_start,
                duration_steps=recovery_duration,
                traffic_multiplier=1.3,   # traffic tails off slowly
                error_multiplier=1.5,     # errors still elevated briefly
                latency_multiplier=1.2,
                event_type="POST_BURST_TAIL",
            )
        )

    pattern = PatternConfig(
        base_traffic=100.0,
        base_error_rate=base_error_rate,
        base_latency_ms=50.0,
        traffic_diurnal_amplitude=0.15,  # modest diurnal during flash sale window
        error_diurnal_amplitude=0.05,
        events=events,
        seed=seed,
    )

    cfg = ScenarioConfig(
        name="flash_sale",
        description=(
            f"Flash sale burst: {traffic_multiplier}× traffic spike at step {spike_start} "
            f"lasting {spike_duration} steps with correlated error surge."
        ),
        total_steps=total_steps,
        pattern=pattern,
        seed=seed,
        # Tighter governance for revenue-critical event
        initial_boundary=0.45,
        beta=0.40,           # fast contraction during spike
        safe_threshold=0.20,
        risk_threshold=0.55,
        decisions_per_step=30,  # higher volume during flash sale
    )
    # Apply caller overrides
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ─── Scenario 2: Degradation ─────────────────────────────────────────────────

def degradation_scenario(
    total_steps: int = 400,
    degradation_start: int = 80,
    final_error_rate: float = 0.45,
    final_latency_ms: float = 250.0,
    base_error_rate: float = 0.05,
    seed: int = 42,
    **overrides,
) -> ScenarioConfig:
    """
    Gradual Degradation scenario.

    Timeline:
      Steps 0–degradation_start  : healthy baseline
      Steps degradation_start–end: linear error + latency increase (no sudden spikes)

    Models: memory leak, connection pool exhaustion, upstream service deterioration.

    Design intent: tests the boundary controller's ability to detect slow-rolling
    degradation before it becomes a hard incident — the "early warning" capability
    from paper §4.1 (confidence–outcome divergence signal).

    Args:
        total_steps: total simulation length
        degradation_start: step at which degradation begins creeping in
        final_error_rate: error rate reached at the end of simulation
        final_latency_ms: latency reached at the end of simulation
        base_error_rate: normal-operation error rate
        seed: RNG seed
        **overrides: any ScenarioConfig field can be overridden
    """
    degradation_steps = total_steps - degradation_start
    if degradation_steps <= 0:
        raise ValueError("degradation_start must be before total_steps")

    # Linear trend per step to reach final value from base
    error_trend = (final_error_rate - base_error_rate) / max(degradation_steps * 100.0, 1)
    latency_trend = (final_latency_ms - 50.0) / max(degradation_steps * 50.0, 1)

    # Single long degradation event (gradual multiplier ramp — we model via trend)
    events: List[EventOverlay] = [
        EventOverlay(
            start_step=degradation_start,
            duration_steps=degradation_steps,
            traffic_multiplier=1.0,        # traffic stays stable
            error_multiplier=1.0,          # trend handles the increase
            latency_multiplier=1.0,
            event_type="DEGRADATION_ONSET",
        )
    ]

    pattern = PatternConfig(
        base_traffic=100.0,
        base_error_rate=base_error_rate,
        base_latency_ms=50.0,
        traffic_diurnal_amplitude=0.25,    # normal diurnal rhythm maintained
        error_diurnal_amplitude=0.08,
        latency_diurnal_amplitude=0.15,
        traffic_noise_sigma=0.04,
        error_noise_sigma=0.015,
        latency_noise_sigma=0.025,
        # Trend drives the gradual increase after degradation_start
        error_trend_per_step=error_trend,
        latency_trend_per_step=latency_trend,
        events=events,
        seed=seed,
    )

    cfg = ScenarioConfig(
        name="degradation",
        description=(
            f"Gradual degradation: error rate climbs from {base_error_rate:.0%} "
            f"to ~{final_error_rate:.0%} over {degradation_steps} steps. "
            "No traffic spike — tests slow-rolling risk detection."
        ),
        total_steps=total_steps,
        pattern=pattern,
        seed=seed,
        initial_boundary=0.5,
        alpha=0.04,          # very slow expansion — degradation needs caution
        beta=0.35,
        safe_threshold=0.25,
        risk_threshold=0.55,
        decisions_per_step=20,
        # Lower latency threshold — slow degradation detected via latency first
        high_latency_threshold_ms=120.0,
    )
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ─── Scenario 3: Recovery ────────────────────────────────────────────────────

def recovery_scenario(
    total_steps: int = 400,
    initial_error_rate: float = 0.50,
    stable_error_rate: float = 0.06,
    recovery_start: int = 0,
    recovery_duration: int = 150,
    seed: int = 42,
    **overrides,
) -> ScenarioConfig:
    """
    Recovery scenario.

    Timeline:
      Steps 0–recovery_duration  : errors fall exponentially from high initial value
      Steps recovery_duration–end: stable, low-error baseline

    Models: post-incident remediation, deployment rollback, circuit breaker opening.

    Design intent: tests the boundary controller's ability to re-expand autonomy
    as the system recovers — the "earned autonomy" expansion property from §3.1.
    Demonstrates that α(1−B) expansion correctly waits for sustained safety evidence.

    Args:
        total_steps: total simulation length
        initial_error_rate: error rate at t=0 (high — system is degraded)
        stable_error_rate: target error rate after recovery completes
        recovery_start: step at which recovery begins (usually 0)
        recovery_duration: how many steps until system is stable
        seed: RNG seed
        **overrides: any ScenarioConfig field can be overridden
    """
    # Start with a high-error event overlay that decays
    # Use a negative error_multiplier trend to model improvement
    error_decay_per_step = (stable_error_rate - initial_error_rate) / max(recovery_duration, 1)
    # This is negative (improving), so error_trend_per_step is negative

    # Post-recovery stability event — no multiplier change, just a marker
    events: List[EventOverlay] = [
        EventOverlay(
            start_step=recovery_start,
            duration_steps=recovery_duration,
            traffic_multiplier=0.8,   # traffic slightly suppressed during recovery
            error_multiplier=1.0,     # trend handles the decrease
            latency_multiplier=1.5,   # latency elevated during recovery operations
            event_type="RECOVERY_IN_PROGRESS",
        ),
        EventOverlay(
            start_step=recovery_start + recovery_duration,
            duration_steps=total_steps - recovery_start - recovery_duration,
            traffic_multiplier=1.0,
            error_multiplier=1.0,
            latency_multiplier=1.0,
            event_type="STABILISED",
        ),
    ]
    # Remove zero-duration events
    events = [e for e in events if e.duration_steps > 0]

    pattern = PatternConfig(
        base_traffic=100.0,
        base_error_rate=initial_error_rate,   # starts HIGH
        base_latency_ms=80.0,                  # starts elevated
        traffic_diurnal_amplitude=0.10,        # low diurnal — system is stressed
        error_diurnal_amplitude=0.05,
        latency_diurnal_amplitude=0.10,
        # Negative error trend drives recovery
        error_trend_per_step=error_decay_per_step / 100.0,
        latency_trend_per_step=-0.003,         # latency improves gradually
        events=events,
        seed=seed,
    )

    cfg = ScenarioConfig(
        name="recovery",
        description=(
            f"Recovery: system starts degraded ({initial_error_rate:.0%} errors) "
            f"and stabilises to {stable_error_rate:.0%} over {recovery_duration} steps. "
            "Tests earned-autonomy re-expansion after incident."
        ),
        total_steps=total_steps,
        pattern=pattern,
        seed=seed,
        # Start with low boundary — system is degraded
        initial_boundary=0.20,
        alpha=0.06,          # allow re-expansion once safe
        beta=0.25,           # moderate contraction (already low)
        safe_threshold=0.20,
        risk_threshold=0.50,
        decisions_per_step=15,  # lower volume during recovery
    )
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ─── Registry ────────────────────────────────────────────────────────────────

SCENARIO_REGISTRY = {
    "flash_sale":  flash_sale_scenario,
    "degradation": degradation_scenario,
    "recovery":    recovery_scenario,
}


def get_scenario(name: str, **kwargs) -> ScenarioConfig:
    """Look up and instantiate a scenario by name."""
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIO_REGISTRY)}")
    return SCENARIO_REGISTRY[name](**kwargs)
