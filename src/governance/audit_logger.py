"""
Audit Logger — structured, append-only audit trail for all governance events.

Every boundary update, routing decision override, human review action, and
shadow mode comparison is recorded here. Supports compliance verification
and post-incident analysis (paper §3.4).

Format: newline-delimited JSON (JSONL) — easy to stream into Splunk/ELK/BigQuery.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    BOUNDARY_UPDATE = "BOUNDARY_UPDATE"
    DECISION_ROUTED = "DECISION_ROUTED"
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"
    HUMAN_REVIEW_COMPLETED = "HUMAN_REVIEW_COMPLETED"
    SHADOW_COMPARISON = "SHADOW_COMPARISON"
    BOUNDARY_FREEZE = "BOUNDARY_FREEZE"
    BOUNDARY_UNFREEZE = "BOUNDARY_UNFREEZE"
    SIMULATION_START = "SIMULATION_START"
    SIMULATION_END = "SIMULATION_END"
    ANOMALY_PERIOD_START = "ANOMALY_PERIOD_START"
    ANOMALY_PERIOD_END = "ANOMALY_PERIOD_END"
    FEEDBACK_SIGNAL_DROPPED = "FEEDBACK_SIGNAL_DROPPED"


class AuditEvent:
    """A single audit record."""
    __slots__ = ("event_type", "timestamp", "actor", "payload")

    def __init__(
        self,
        event_type: AuditEventType,
        payload: Dict[str, Any],
        actor: str = "system",
    ) -> None:
        self.event_type = event_type
        self.timestamp = datetime.utcnow().isoformat()
        self.actor = actor
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "actor": self.actor,
            **self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """
    Thread-safe append-only audit logger.

    Writes to a JSONL file and optionally to an in-memory ring buffer for
    the dashboard/API to query recent events without filesystem reads.
    """

    def __init__(
        self,
        log_path: str = "logs/audit.jsonl",
        ring_buffer_size: int = 1000,
        enabled: bool = True,
    ) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ring: list[dict] = []
        self._ring_size = ring_buffer_size
        self._lock = threading.Lock()
        self._enabled = enabled
        self._count = 0

    def log(
        self,
        event_type: AuditEventType,
        payload: Dict[str, Any],
        actor: str = "system",
    ) -> None:
        if not self._enabled:
            return
        event = AuditEvent(event_type=event_type, payload=payload, actor=actor)
        record = event.to_dict()
        with self._lock:
            # Write to JSONL file
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(event.to_json() + "\n")
            except OSError as e:
                logger.error("Audit log write failed: %s", e)

            # Maintain ring buffer
            self._ring.append(record)
            if len(self._ring) > self._ring_size:
                self._ring.pop(0)
            self._count += 1

    def recent_events(self, n: int = 50) -> list[dict]:
        with self._lock:
            return list(self._ring[-n:])

    def event_count(self) -> int:
        with self._lock:
            return self._count

    def log_boundary_update(self, update_result: Any) -> None:
        self.log(AuditEventType.BOUNDARY_UPDATE, update_result.to_dict())

    def log_human_override(
        self,
        decision_id: str,
        decision_type: str,
        operator: str,
        reason: str,
        original_outcome: str,
        new_outcome: str,
    ) -> None:
        self.log(
            AuditEventType.HUMAN_OVERRIDE,
            {
                "decision_id": decision_id,
                "decision_type": decision_type,
                "original_outcome": original_outcome,
                "new_outcome": new_outcome,
                "reason": reason,
            },
            actor=operator,
        )

    def log_anomaly_start(self, step: int, multiplier: float) -> None:
        self.log(
            AuditEventType.ANOMALY_PERIOD_START,
            {"step": step, "risk_multiplier": multiplier},
        )

    def log_anomaly_end(self, step: int) -> None:
        self.log(AuditEventType.ANOMALY_PERIOD_END, {"step": step})
