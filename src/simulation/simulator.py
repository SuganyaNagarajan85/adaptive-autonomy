"""
Simulator — orchestrates one full run of the Autonomy Learning Loop.

Drives the complete feedback cycle per time step:
  1. Generate traffic (TrafficGenerator)
  2. Route each decision (AutonomyController)
  3. Execute or queue for review (DecisionEngine)
  4. Simulate outcomes (probabilistic, conditioned on anomaly state)
  5. Emit feedback signals (FeedbackCollector)
  6. At window boundary: aggregate signals (FeedbackAggregator)
  7. Update boundaries (BoundaryUpdater → AutonomyController)
  8. Record metrics (MetricsCollector)
  9. Log audit events (AuditLogger)

Two modes:
  - adaptive: full feedback loop active (B evolves)
  - static:   B fixed at initial value (baseline comparison)
"""
from __future__ import annotations

import logging
import random
import time
from typing import Dict, List, Optional

from src.autonomy_controller.controller import AutonomyController
from src.boundary_update.updater import BoundaryUpdater
from src.decision_engine.engine import DecisionEngine
from src.decision_engine.models import DecisionOutcome, DecisionType, ExecutionMode
from src.experiments.metrics import MetricsCollector, StepMetrics
from src.feedback_aggregator.aggregator import FeedbackAggregator
from src.feedback_signals.collector import FeedbackCollector
from src.feedback_signals.models import SignalSeverity, SignalType
from src.governance.audit_logger import AuditEventType, AuditLogger
from src.governance.human_review import HumanReviewQueue
from src.governance.shadow_mode import ShadowModeEngine
from src.simulation.noise_injector import NoiseInjector
from src.simulation.traffic_generator import TrafficGenerator

logger = logging.getLogger(__name__)

# Forward-declared to avoid a hard circular import; TYPE_CHECKING block handles type hints.
# The actual import is deferred inside __init__ to keep startup cost zero when the
# production simulator is not used.
_PRODUCTION_SIM_TYPE = None


class Simulator:
    """
    Full simulation engine for the Autonomy Learning Loop.
    """

    def __init__(
        self,
        total_steps: int = 500,
        decisions_per_step: int = 20,
        base_failure_rate: float = 0.08,
        anomaly_probability: float = 0.05,
        anomaly_duration_steps: int = 10,
        peak_multiplier: float = 3.5,
        alpha: float = 0.05,
        beta: float = 0.30,
        safe_threshold: float = 0.25,
        risk_threshold: float = 0.60,
        initial_boundary: float = 0.5,
        min_boundary: float = 0.05,
        max_boundary: float = 0.95,
        window_size: int = 50,
        headroom_buffer: float = 0.15,
        spike_growth_threshold: float = 1.5,
        spike_consecutive_windows: int = 2,
        spike_conservative_boundary: float = 0.15,
        spike_recovery_lock_windows: int = 10,
        sustained_safe_windows: int = 3,
        spike_trend_window_count: int = 4,
        spike_trend_slope_threshold: float = 0.02,
        min_update_interval_seconds: float = 0.0,  # 0 = no rate limit in simulation
        shadow_mode: bool = False,
        adaptive: bool = True,
        seed: int = 42,
        audit_log_path: str = "logs/audit.jsonl",
        class_overrides: Optional[Dict[str, float]] = None,
        signal_weights: Optional[dict] = None,
        # ── Production simulator hook (optional) ──────────────────────────
        # When set, replaces the NoiseInjector + base_failure_rate pair.
        # Must expose a .step(t) -> ProductionEnvState interface.
        production_sim=None,
    ) -> None:
        self.total_steps = total_steps
        self.decisions_per_step = decisions_per_step
        self.base_failure_rate = base_failure_rate
        self.adaptive = adaptive
        self._seed = seed
        self._rng = random.Random(seed)

        # Core components
        self.traffic_gen = TrafficGenerator(decisions_per_step, seed=seed)
        self.noise_injector = NoiseInjector(
            anomaly_probability=anomaly_probability,
            anomaly_duration_steps=anomaly_duration_steps,
            peak_risk_multiplier=peak_multiplier,
            seed=seed + 1,
        )
        self.decision_engine = DecisionEngine(seed=seed + 2)
        self.controller = AutonomyController(
            initial_boundary=initial_boundary,
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            shadow_mode=shadow_mode,
            seed=seed + 3,
            class_overrides=class_overrides,
        )
        self.feedback_collector = FeedbackCollector(max_size=50_000)
        self.aggregator = FeedbackAggregator(
            window_size=window_size,
            signal_weights=signal_weights,
        )
        self.boundary_updater = BoundaryUpdater(
            alpha=alpha,
            beta=beta,
            safe_threshold=safe_threshold,
            risk_threshold=risk_threshold,
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            min_update_interval_seconds=min_update_interval_seconds,
            headroom_buffer=headroom_buffer,
            spike_growth_threshold=spike_growth_threshold,
            spike_consecutive_windows=spike_consecutive_windows,
            spike_conservative_boundary=spike_conservative_boundary,
            spike_recovery_lock_windows=spike_recovery_lock_windows,
            sustained_safe_windows=sustained_safe_windows,
            spike_trend_window_count=spike_trend_window_count,
            spike_trend_slope_threshold=spike_trend_slope_threshold,
        )
        self.review_queue = HumanReviewQueue(sla_seconds=300)
        self.shadow_engine = ShadowModeEngine()
        self.audit_logger = AuditLogger(log_path=audit_log_path)
        self.metrics_collector = MetricsCollector()
        # Optional production simulator — overrides noise_injector when set
        self.production_sim = production_sim

    def run(self) -> List[StepMetrics]:
        """
        Execute the full simulation. Returns per-step metrics list.
        """
        self.audit_logger.log(
            AuditEventType.SIMULATION_START,
            {"total_steps": self.total_steps, "adaptive": self.adaptive, "seed": self._seed},
        )
        logger.info(
            "Simulation starting: steps=%d adaptive=%s seed=%d",
            self.total_steps, self.adaptive, self._seed,
        )

        prev_anomaly_active = False
        step_metrics_list: List[StepMetrics] = []

        for step in range(self.total_steps):
            # ── Environment state: production sim OR classic noise injector ──
            if self.production_sim is not None:
                anomaly = self.production_sim.step(step)
                # Emit environment-level signals once per step (in addition to
                # per-decision outcome signals generated below).
                from src.simulation.production_simulator import map_env_to_signals
                for dt in DecisionType:
                    env_signals = map_env_to_signals(
                        anomaly,
                        self.production_sim.scenario,
                        decision_type=dt.value,
                    )
                    if env_signals:
                        self.feedback_collector.emit_batch(env_signals)
            else:
                anomaly = self.noise_injector.step()

            # Audit anomaly transitions
            if anomaly.active and not prev_anomaly_active:
                self.audit_logger.log_anomaly_start(step, anomaly.risk_multiplier)
                logger.debug("Anomaly period started at step %d: %s", step, anomaly.anomaly_type)
            elif not anomaly.active and prev_anomaly_active:
                self.audit_logger.log_anomaly_end(step)
                logger.debug("Anomaly period ended at step %d", step)
            prev_anomaly_active = anomaly.active

            # Generate traffic for this step
            requests = self.traffic_gen.generate_step(
                traffic_multiplier=anomaly.traffic_multiplier
            )

            step_auto = 0
            step_human = 0
            step_incidents = 0
            step_rollbacks = 0
            step_overrides = 0
            step_cb_fired = False

            # Boundary snapshots at step start for metrics
            boundary_snapshots: Dict[str, float] = {
                dt.value: self.controller.get_boundary(dt) for dt in DecisionType
            }

            # --- Process each decision ---
            for dt, user_id, session_id in requests:
                routing = self.controller.route(dt)
                b = routing.boundary_at_routing

                decision = self.decision_engine.make_decision(
                    decision_type=dt,
                    autonomy_boundary=b,
                    is_anomaly_period=anomaly.active,
                    traffic_multiplier=anomaly.traffic_multiplier,
                    execution_mode=routing.execution_mode,
                    user_id=user_id,
                    session_id=session_id,
                )

                if routing.execution_mode == ExecutionMode.HUMAN_REVIEW:
                    step_human += 1
                    self.review_queue.enqueue_by_type(dt, b, decision.decision_id)
                    decision.resolve_outcome(DecisionOutcome.SUCCESS)
                else:
                    step_auto += 1
                    # Production mode uses measured error_rate directly as P(failure)
                    prod_failure_rate = (
                        getattr(anomaly, "derived_failure_rate", None)
                        if self.production_sim is not None else None
                    )
                    outcome = self._simulate_outcome(
                        confidence=decision.context.confidence_score,
                        is_anomaly=anomaly.active,
                        risk_multiplier=anomaly.risk_multiplier,
                        decision_type=dt,
                        override_failure_rate=prod_failure_rate,
                    )
                    decision.resolve_outcome(outcome)

                    # Emit feedback signals based on outcome
                    signals = self._generate_signals(decision, anomaly.active)
                    self.feedback_collector.emit_batch(signals)

                    # Record for aggregator
                    self.aggregator.record_decision(
                        dt.value,
                        is_auto=True,
                        outcome=outcome.value,
                    )

                    if outcome == DecisionOutcome.ROLLBACK:
                        step_rollbacks += 1
                    elif outcome == DecisionOutcome.OVERRIDE:
                        step_overrides += 1
                    elif outcome == DecisionOutcome.INCIDENT:
                        step_incidents += 1

                    # Shadow mode comparison
                    if routing.execution_mode == ExecutionMode.SHADOW:
                        human_outcome = self._simulate_human_outcome(outcome)
                        self.shadow_engine.record(
                            decision.decision_id,
                            dt.value,
                            outcome.value,
                            human_outcome,
                            decision.context.confidence_score,
                            b,
                        )

            # --- Window boundary: aggregate & update ---
            signals_batch = self.feedback_collector.drain()
            self.aggregator.ingest_batch(signals_batch)
            window_metrics = self.aggregator.evaluate_window()

            # Apply boundary updates if adaptive
            update_results = {}
            if self.adaptive:
                for dt in DecisionType:
                    wm = window_metrics.get(dt.value)
                    if wm is None:
                        continue
                    current_b = self.controller.get_boundary(dt)
                    result = self.boundary_updater.compute_update(
                        current_b,
                        wm,
                        is_frozen=self.controller.is_frozen(dt),
                    )
                    if result.delta != 0.0 or result.circuit_breaker_triggered:
                        self.controller.set_boundary(dt, result.boundary_after)
                        self.audit_logger.log_boundary_update(result)
                        update_results[dt.value] = result.to_dict()
                    if result.circuit_breaker_triggered:
                        step_cb_fired = True

            # Collect step metrics
            # successful_auto: auto decisions with no adverse outcome.
            # Human-review decisions always resolve SUCCESS, so all adverse
            # outcomes belong exclusively to the autonomous path — exact count.
            step_successful_auto = max(
                0, step_auto - step_incidents - step_rollbacks - step_overrides
            )
            sm = StepMetrics(
                step=step,
                is_anomaly=anomaly.active,
                anomaly_type=anomaly.anomaly_type,
                total_decisions=step_auto + step_human,
                auto_decisions=step_auto,
                human_review_decisions=step_human,
                incidents=step_incidents,
                rollbacks=step_rollbacks,
                overrides=step_overrides,
                boundary_snapshots=boundary_snapshots,
                window_metrics={k: v.to_dict() for k, v in window_metrics.items()},
                review_queue_depth=self.review_queue.queue_depth(),
                successful_auto_decisions=step_successful_auto,
                circuit_breaker_fired=step_cb_fired,
            )
            self.metrics_collector.record(sm)
            step_metrics_list.append(sm)

            # Expire stale review items periodically
            if step % 50 == 0:
                self.review_queue.expire_stale()

        self.audit_logger.log(
            AuditEventType.SIMULATION_END,
            {"steps_completed": self.total_steps, "adaptive": self.adaptive},
        )
        logger.info("Simulation complete: %d steps", self.total_steps)
        return step_metrics_list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_outcome(
        self,
        confidence: float,
        is_anomaly: bool,
        risk_multiplier: float,
        decision_type: DecisionType,
        override_failure_rate: Optional[float] = None,
    ) -> DecisionOutcome:
        """
        Probabilistically determine a decision outcome.

        When override_failure_rate is provided (production mode), it is used
        directly as P(failure) instead of computing from base_failure_rate.
        Confidence still modulates the production failure rate to model the
        effect of a well-calibrated decision model reducing failures.
        """
        if override_failure_rate is not None:
            # Production mode: error_rate × confidence-reduction factor
            p_fail = min(0.95, override_failure_rate * (1.0 - confidence * 0.3))
        else:
            # Classic mode: base failure rate modulated by confidence and anomaly state
            p_fail = self.base_failure_rate * (1.0 - confidence * 0.5)
            if is_anomaly:
                p_fail = min(0.85, p_fail * risk_multiplier)

        draw = self._rng.random()
        if draw > p_fail:
            return DecisionOutcome.SUCCESS

        # Distribute failures across adverse outcome types
        fail_draw = self._rng.random()
        if fail_draw < 0.40:
            return DecisionOutcome.ROLLBACK
        elif fail_draw < 0.70:
            return DecisionOutcome.OVERRIDE
        else:
            return DecisionOutcome.INCIDENT

    def _simulate_human_outcome(self, auto_outcome: DecisionOutcome) -> str:
        """
        Shadow mode: what would a human reviewer have decided?
        Humans catch ~80% of adverse outcomes that auto misses.
        """
        if auto_outcome == DecisionOutcome.SUCCESS:
            # Humans occasionally disagree with even good decisions (5%)
            return "OVERRIDE" if self._rng.random() < 0.05 else "SUCCESS"
        else:
            # Humans catch most failures
            return "SUCCESS" if self._rng.random() < 0.80 else auto_outcome.value

    def _generate_signals(self, decision, is_anomaly: bool) -> list:
        """Generate appropriate feedback signals based on decision outcome."""
        from src.feedback_signals.collector import FeedbackCollector as FC
        signals = []

        if decision.outcome == DecisionOutcome.ROLLBACK:
            severity = SignalSeverity.HIGH if is_anomaly else SignalSeverity.MEDIUM
            signals.append(FC.make_rollback(
                decision.decision_id,
                decision.decision_type.value,
                recovery_cost_seconds=self._rng.uniform(15, 120),
                severity=severity,
            ))

        elif decision.outcome == DecisionOutcome.OVERRIDE:
            signals.append(FC.make_override(
                decision.decision_id,
                decision.decision_type.value,
                severity=SignalSeverity.HIGH if is_anomaly else SignalSeverity.MEDIUM,
            ))

        elif decision.outcome == DecisionOutcome.INCIDENT:
            signals.append(FC.make_incident(
                decision.decision_id,
                decision.decision_type.value,
                severity=SignalSeverity.CRITICAL if is_anomaly else SignalSeverity.HIGH,
            ))

        # Confidence divergence: high confidence but bad outcome
        if decision.is_adverse and decision.context.confidence_score > 0.75:
            signals.append(FC.make_confidence_divergence(
                decision.decision_id,
                decision.decision_type.value,
                confidence_at_decision=decision.context.confidence_score,
            ))

        # Trust degradation during sustained anomaly periods (probabilistic)
        if is_anomaly and self._rng.random() < 0.15:
            signals.append(FC.make_trust_degradation(
                decision.decision_type.value,
                severity=SignalSeverity.MEDIUM,
            ))

        return signals
