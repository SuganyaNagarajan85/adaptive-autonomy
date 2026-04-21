"""
Feedback Signal Collector — async, non-blocking signal ingestion.

Signals arrive from multiple production systems (order service, incident manager,
override UI, trust/safety pipeline) and are buffered in a thread-safe queue.
The aggregator drains this queue on each window boundary.

Design principle (paper §3.2):
    "Outcome capture is performed asynchronously and decoupled from request
     execution paths to avoid impacting decision latency and system throughput."
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import List, Optional

from .models import FeedbackSignal, SignalSeverity, SignalType

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Thread-safe in-process signal buffer. In production this would be backed by
    a Kafka consumer or SQS queue; here we use a bounded deque for simulation fidelity.

    Back-pressure: when the queue is full, signals are dropped with a WARNING log
    rather than blocking the caller — keeping the hot path safe.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._queue: queue.Queue[FeedbackSignal] = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._dropped_count = 0
        self._total_received = 0

    def emit(self, signal: FeedbackSignal) -> bool:
        """
        Non-blocking emit. Returns True if enqueued, False if dropped (back-pressure).
        Called from any thread; safe for concurrent producers.
        """
        try:
            self._queue.put_nowait(signal)
            with self._lock:
                self._total_received += 1
            return True
        except queue.Full:
            with self._lock:
                self._dropped_count += 1
            logger.warning(
                "FeedbackCollector queue full — signal dropped",
                extra={"signal_type": signal.signal_type.value, "dropped_total": self._dropped_count},
            )
            return False

    def emit_batch(self, signals: List[FeedbackSignal]) -> int:
        """Emit multiple signals; returns count successfully enqueued."""
        return sum(1 for s in signals if self.emit(s))

    def drain(self, max_items: Optional[int] = None) -> List[FeedbackSignal]:
        """
        Drain up to max_items signals from the queue (non-blocking).
        Called by the aggregator at window boundaries.
        """
        drained: List[FeedbackSignal] = []
        limit = max_items or self._queue.maxsize
        while len(drained) < limit:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return drained

    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "total_received": self._total_received,
                "dropped": self._dropped_count,
                "queue_depth": self._queue.qsize(),
            }

    # ------------------------------------------------------------------
    # Convenience factory methods for common signal types
    # ------------------------------------------------------------------

    @staticmethod
    def make_rollback(
        decision_id: str,
        decision_type: str,
        recovery_cost_seconds: float = 30.0,
        severity: SignalSeverity = SignalSeverity.HIGH,
    ) -> FeedbackSignal:
        return FeedbackSignal(
            signal_type=SignalType.ROLLBACK,
            severity=severity,
            decision_id=decision_id,
            decision_type=decision_type,
            description=f"Decision {decision_id} rolled back",
            recovery_cost_seconds=recovery_cost_seconds,
        )

    @staticmethod
    def make_override(
        decision_id: str,
        decision_type: str,
        severity: SignalSeverity = SignalSeverity.MEDIUM,
    ) -> FeedbackSignal:
        return FeedbackSignal(
            signal_type=SignalType.OVERRIDE,
            severity=severity,
            decision_id=decision_id,
            decision_type=decision_type,
            description=f"Human override on {decision_id}",
        )

    @staticmethod
    def make_incident(
        decision_id: str,
        decision_type: str,
        severity: SignalSeverity = SignalSeverity.HIGH,
    ) -> FeedbackSignal:
        return FeedbackSignal(
            signal_type=SignalType.INCIDENT,
            severity=severity,
            decision_id=decision_id,
            decision_type=decision_type,
            description=f"Incident correlated with decision {decision_id}",
        )

    @staticmethod
    def make_trust_degradation(
        decision_type: str,
        severity: SignalSeverity = SignalSeverity.MEDIUM,
    ) -> FeedbackSignal:
        return FeedbackSignal(
            signal_type=SignalType.TRUST_DEGRADATION,
            severity=severity,
            decision_type=decision_type,
            description="Trust degradation signal: elevated abandonment/opt-outs",
        )

    @staticmethod
    def make_confidence_divergence(
        decision_id: str,
        decision_type: str,
        confidence_at_decision: float,
        severity: SignalSeverity = SignalSeverity.MEDIUM,
    ) -> FeedbackSignal:
        return FeedbackSignal(
            signal_type=SignalType.CONFIDENCE_DIVERGENCE,
            severity=severity,
            decision_id=decision_id,
            decision_type=decision_type,
            description=f"High confidence ({confidence_at_decision:.2f}) but adverse outcome",
            confidence_at_decision=confidence_at_decision,
        )
