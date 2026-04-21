"""
FastAPI application — production REST API for the Autonomy Learning Loop.

Endpoints:
  POST /decisions/route          — route a decision through the controller
  GET  /decisions/boundaries     — all boundary states
  GET  /decisions/boundaries/{t} — single class boundary
  GET  /overrides/queue          — pending human review items
  POST /overrides/resolve        — resolve a review item
  GET  /overrides/queue/stats    — queue statistics
  GET  /metrics/audit/recent     — recent audit events
  GET  /health                   — liveness probe

Run:  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import decisions, metrics, overrides
from src.autonomy_controller.controller import AutonomyController
from src.governance.audit_logger import AuditLogger
from src.governance.human_review import HumanReviewQueue


def load_config(config_path: str = "config/ecommerce.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared system components on startup."""
    cfg = load_config()

    autonomy_cfg = cfg.get("autonomy", {})
    gov_cfg = cfg.get("governance", {})

    # Build shared components
    class_overrides = {}
    for dt_name, overrides_cfg in cfg.get("decision_class_overrides", {}).items():
        class_overrides[dt_name] = overrides_cfg.get(
            "initial_boundary", autonomy_cfg.get("initial_boundary", 0.5)
        )

    controller = AutonomyController(
        initial_boundary=autonomy_cfg.get("initial_boundary", 0.5),
        min_boundary=autonomy_cfg.get("min_boundary", 0.05),
        max_boundary=autonomy_cfg.get("max_boundary", 0.95),
        shadow_mode=gov_cfg.get("shadow_mode_enabled", False),
        class_overrides=class_overrides,
    )

    review_queue = HumanReviewQueue(
        capacity=gov_cfg.get("human_review_queue_capacity", 200),
        sla_seconds=gov_cfg.get("review_sla_seconds", 300),
    )

    audit_logger = AuditLogger(
        log_path=gov_cfg.get("audit_log_path", "logs/audit.jsonl"),
    )

    # Wire into route modules
    decisions.set_controller(controller)
    overrides.set_review_queue(review_queue)
    metrics.set_audit_logger(audit_logger)

    # Store in app state for access by other code
    app.state.controller = controller
    app.state.review_queue = review_queue
    app.state.audit_logger = audit_logger

    yield
    # Cleanup (if needed)


app = FastAPI(
    title="Feedback-Driven Autonomy API",
    description=(
        "Production REST API for the Autonomy Learning Loop — "
        "adaptive governance of autonomous decision systems."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(decisions.router)
app.include_router(overrides.router)
app.include_router(metrics.router)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "autonomy-learning-loop"}


@app.get("/")
def root() -> Dict[str, str]:
    return {
        "message": "Feedback-Driven Autonomy API",
        "docs": "/docs",
        "redoc": "/redoc",
    }
