"""
Human override and review queue API routes.
Provides the backend for the human-in-the-loop UI.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.governance.human_review import HumanReviewQueue, ReviewStatus

router = APIRouter(prefix="/overrides", tags=["overrides"])

# Global review queue reference — set by app startup
_review_queue: HumanReviewQueue | None = None


def set_review_queue(queue: HumanReviewQueue) -> None:
    global _review_queue
    _review_queue = queue


def get_review_queue() -> HumanReviewQueue:
    if _review_queue is None:
        raise HTTPException(status_code=503, detail="Review queue not initialized")
    return _review_queue


class ResolveRequest(BaseModel):
    item_id: str
    status: ReviewStatus
    reviewer: str = "operator"
    note: str = ""


@router.get("/queue", response_model=List[Dict[str, Any]])
def get_pending_reviews() -> List[Dict[str, Any]]:
    """Return all pending human review items."""
    return get_review_queue().pending_items()


@router.get("/queue/stats")
def get_queue_stats() -> Dict[str, Any]:
    """Review queue statistics."""
    return get_review_queue().stats


@router.post("/resolve")
def resolve_review(request: ResolveRequest) -> Dict[str, Any]:
    """Resolve a review item (approve or override)."""
    queue = get_review_queue()
    item = queue.resolve(
        item_id=request.item_id,
        status=request.status,
        reviewer=request.reviewer,
        note=request.note,
    )
    if item is None:
        raise HTTPException(status_code=404, detail=f"Review item {request.item_id} not found")
    return {"resolved": True, "item": item.to_dict()}


@router.post("/expire-stale")
def expire_stale() -> Dict[str, int]:
    """Expire review items past SLA deadline."""
    count = get_review_queue().expire_stale()
    return {"expired_count": count}
