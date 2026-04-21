"""
Metrics and audit API routes — exposes system state for dashboards.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from src.governance.audit_logger import AuditLogger

router = APIRouter(prefix="/metrics", tags=["metrics"])

_audit_logger: AuditLogger | None = None


def set_audit_logger(logger: AuditLogger) -> None:
    global _audit_logger
    _audit_logger = logger


@router.get("/audit/recent")
def recent_audit_events(n: int = 50) -> List[Dict[str, Any]]:
    """Return last N audit events."""
    if _audit_logger is None:
        return []
    return _audit_logger.recent_events(n)


@router.get("/audit/count")
def audit_event_count() -> Dict[str, int]:
    if _audit_logger is None:
        return {"count": 0}
    return {"count": _audit_logger.event_count()}
