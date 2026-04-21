"""
Human Review Queue — manages decisions routed to human operators.

In e-commerce: fraud alerts, high-value offer approvals, pricing anomalies
that exceed the confidence threshold land here. The queue is prioritized by
decision type severity and SLA deadline. This is the "human-in-the-loop UI"
backend (paper §3.4 + additional features).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

from src.decision_engine.models import Decision, DecisionType


# Priority scoring per decision type — higher = reviewed first
_DECISION_PRIORITY: Dict[DecisionType, int] = {
    DecisionType.FRAUD_DETECTION: 10,
    DecisionType.PRICING: 7,
    DecisionType.OFFER_SELECTION: 6,
    DecisionType.NOTIFICATION: 4,
    DecisionType.PRODUCT_RECOMMENDATION: 3,
    DecisionType.SEARCH_RANKING: 2,
}


class ReviewStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"     # human approved the auto decision
    OVERRIDDEN = "OVERRIDDEN"  # human changed the decision
    EXPIRED = "EXPIRED"       # SLA deadline passed — auto-approved with warning
    ESCALATED = "ESCALATED"   # routed to senior operator


@dataclass(order=True)
class ReviewItem:
    """A decision queued for human review. Sortable by priority (highest first)."""
    priority: int
    item_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    decision: Optional[Decision] = field(compare=False, default=None)
    decision_type: str = field(compare=False, default="")
    boundary_at_routing: float = field(compare=False, default=0.0)
    queued_at: datetime = field(compare=False, default_factory=datetime.utcnow)
    sla_deadline: Optional[datetime] = field(compare=False, default=None)
    status: ReviewStatus = field(compare=False, default=ReviewStatus.PENDING)
    reviewer: Optional[str] = field(compare=False, default=None)
    review_note: str = field(compare=False, default="")
    resolved_at: Optional[datetime] = field(compare=False, default=None)

    @property
    def is_expired(self) -> bool:
        if self.sla_deadline is None:
            return False
        return datetime.utcnow() > self.sla_deadline

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.queued_at).total_seconds()

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "decision_type": self.decision_type,
            "priority": self.priority,
            "status": self.status.value,
            "queued_at": self.queued_at.isoformat(),
            "sla_deadline": self.sla_deadline.isoformat() if self.sla_deadline else None,
            "age_seconds": round(self.age_seconds, 1),
            "is_expired": self.is_expired,
            "reviewer": self.reviewer,
            "review_note": self.review_note,
        }


class HumanReviewQueue:
    """
    Priority queue for human review items with SLA tracking.

    Thread-safe: designed to be accessed from both the simulation thread
    (enqueue) and the API/UI thread (dequeue, resolve).
    """

    def __init__(
        self,
        capacity: int = 200,
        sla_seconds: int = 300,
    ) -> None:
        self._capacity = capacity
        self._sla_seconds = sla_seconds
        self._items: List[ReviewItem] = []
        self._lock = threading.Lock()
        self._total_enqueued = 0
        self._total_resolved = 0
        self._total_expired = 0
        self._overrides: List[dict] = []

    def enqueue(self, decision: Decision, boundary: float) -> Optional[ReviewItem]:
        """Add a decision to the review queue. Returns None if queue is at capacity."""
        with self._lock:
            if len(self._items) >= self._capacity:
                return None

            priority = _DECISION_PRIORITY.get(decision.decision_type, 1)
            # Negative for descending sort (highest priority first)
            item = ReviewItem(
                priority=-priority,
                decision=decision,
                decision_type=decision.decision_type.value,
                boundary_at_routing=boundary,
                sla_deadline=datetime.utcnow() + timedelta(seconds=self._sla_seconds),
            )
            self._items.append(item)
            self._items.sort()
            self._total_enqueued += 1
            return item

    def enqueue_by_type(
        self,
        decision_type: DecisionType,
        boundary: float,
        decision_id: str = "",
    ) -> Optional[ReviewItem]:
        """Lightweight enqueue without a full Decision object (for simulation)."""
        with self._lock:
            if len(self._items) >= self._capacity:
                return None
            priority = _DECISION_PRIORITY.get(decision_type, 1)
            item = ReviewItem(
                priority=-priority,
                decision_type=decision_type.value,
                boundary_at_routing=boundary,
                sla_deadline=datetime.utcnow() + timedelta(seconds=self._sla_seconds),
            )
            self._items.append(item)
            self._items.sort()
            self._total_enqueued += 1
            return item

    def resolve(
        self,
        item_id: str,
        status: ReviewStatus,
        reviewer: str = "operator",
        note: str = "",
    ) -> Optional[ReviewItem]:
        """Mark a review item as resolved."""
        with self._lock:
            for item in self._items:
                if item.item_id == item_id:
                    item.status = status
                    item.reviewer = reviewer
                    item.review_note = note
                    item.resolved_at = datetime.utcnow()
                    self._items.remove(item)
                    self._total_resolved += 1
                    if status == ReviewStatus.OVERRIDDEN:
                        self._overrides.append(item.to_dict())
                    return item
        return None

    def expire_stale(self) -> int:
        """Auto-expire items past SLA deadline. Returns count expired."""
        expired_count = 0
        with self._lock:
            to_expire = [i for i in self._items if i.is_expired]
            for item in to_expire:
                item.status = ReviewStatus.EXPIRED
                self._items.remove(item)
                self._total_expired += 1
                expired_count += 1
        return expired_count

    def pending_items(self) -> List[dict]:
        with self._lock:
            return [i.to_dict() for i in self._items if i.status == ReviewStatus.PENDING]

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "queue_depth": len(self._items),
                "capacity": self._capacity,
                "total_enqueued": self._total_enqueued,
                "total_resolved": self._total_resolved,
                "total_expired": self._total_expired,
                "override_count": len(self._overrides),
                "override_rate": round(
                    len(self._overrides) / max(self._total_resolved, 1), 4
                ),
            }
