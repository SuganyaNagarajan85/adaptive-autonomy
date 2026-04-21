"""
Decision routing API routes.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.decision_engine.models import DecisionType
from src.autonomy_controller.controller import AutonomyController

router = APIRouter(prefix="/decisions", tags=["decisions"])


class RouteRequest(BaseModel):
    decision_type: DecisionType
    user_id: str = "anonymous"
    session_id: str = ""


class RouteResponse(BaseModel):
    decision_type: str
    outcome: str
    execution_mode: str
    boundary_at_routing: float
    reason: str


# Dependency injection placeholder — real app wires this in app state
_controller: AutonomyController | None = None


def get_controller() -> AutonomyController:
    if _controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    return _controller


def set_controller(controller: AutonomyController) -> None:
    global _controller
    _controller = controller


@router.post("/route", response_model=RouteResponse)
def route_decision(
    request: RouteRequest,
    controller: AutonomyController = Depends(get_controller),
) -> RouteResponse:
    """Route a single decision through the autonomy controller."""
    routing = controller.route(request.decision_type)
    return RouteResponse(
        decision_type=routing.decision_type.value,
        outcome=routing.outcome.value,
        execution_mode=routing.execution_mode.value,
        boundary_at_routing=routing.boundary_at_routing,
        reason=routing.reason or "",
    )


@router.get("/boundaries")
def get_all_boundaries(
    controller: AutonomyController = Depends(get_controller),
) -> Dict[str, Any]:
    """Return current autonomy boundary for all decision classes."""
    return controller.all_states()


@router.get("/boundaries/{decision_type}")
def get_boundary(
    decision_type: DecisionType,
    controller: AutonomyController = Depends(get_controller),
) -> Dict[str, Any]:
    """Return boundary state for a specific decision class."""
    return controller.get_state(decision_type).snapshot()
