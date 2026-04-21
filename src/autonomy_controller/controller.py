"""
Autonomy Controller — the central governance gate.

Maintains one AutonomyState (boundary B) per decision class and probabilistically
routes each incoming decision to AUTO_EXECUTE, HUMAN_REVIEW, or SHADOW_EXECUTE
based on the current B value.

Thread-safe: uses per-class locks so high-frequency decision types don't contend.
"""
from __future__ import annotations

import random
import threading
from typing import Dict, Optional

from src.decision_engine.models import DecisionType, ExecutionMode
from .models import AutonomyState, RoutingDecision, RoutingOutcome


class AutonomyController:
    """
    Maintains per-class autonomy boundaries and routes decisions.

    Routing logic (probabilistic delegation):
        draw u ~ Uniform(0, 1)
        if shadow_mode and u < B:   → SHADOW_EXECUTE  (auto + human comparison)
        elif u < B:                 → AUTO_EXECUTE
        else:                       → HUMAN_REVIEW

    The boundary B represents the fraction of decisions executed autonomously.
    As B → 1 more decisions are automated; as B → 0 more go to humans.
    """

    def __init__(
        self,
        initial_boundary: float = 0.5,
        min_boundary: float = 0.05,
        max_boundary: float = 0.95,
        shadow_mode: bool = False,
        seed: Optional[int] = None,
        class_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self._initial_boundary = initial_boundary
        self._min_boundary = min_boundary
        self._max_boundary = max_boundary
        self._shadow_mode = shadow_mode
        self._rng = random.Random(seed)
        self._class_overrides: Dict[str, float] = class_overrides or {}

        # Per-class state and per-class lock for thread safety
        self._states: Dict[DecisionType, AutonomyState] = {}
        self._locks: Dict[DecisionType, threading.Lock] = {}
        for dt in DecisionType:
            initial = self._class_overrides.get(dt.value, initial_boundary)
            self._states[dt] = AutonomyState(decision_type=dt, boundary=initial)
            self._locks[dt] = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, decision_type: DecisionType) -> RoutingDecision:
        """
        Probabilistically route a decision based on current boundary B.
        Called on the hot path — must be fast (no I/O).
        """
        with self._locks[decision_type]:
            state = self._states[decision_type]
            b = state.boundary

        draw = self._rng.random()

        if draw < b:
            if self._shadow_mode:
                outcome = RoutingOutcome.SHADOW_EXECUTE
                mode = ExecutionMode.SHADOW
                reason = f"B={b:.3f}, draw={draw:.3f} < B → shadow execute"
            else:
                outcome = RoutingOutcome.AUTO_EXECUTE
                mode = ExecutionMode.AUTO
                reason = f"B={b:.3f}, draw={draw:.3f} < B → auto execute"
        else:
            outcome = RoutingOutcome.HUMAN_REVIEW
            mode = ExecutionMode.HUMAN_REVIEW
            reason = f"B={b:.3f}, draw={draw:.3f} >= B → human review"

        return RoutingDecision(
            decision_type=decision_type,
            boundary_at_routing=b,
            outcome=outcome,
            execution_mode=mode,
            random_draw=draw,
            reason=reason,
        )

    def get_boundary(self, decision_type: DecisionType) -> float:
        with self._locks[decision_type]:
            return self._states[decision_type].boundary

    def set_boundary(self, decision_type: DecisionType, new_value: float) -> None:
        """Called by BoundaryUpdater after each window evaluation."""
        clamped = max(self._min_boundary, min(self._max_boundary, new_value))
        with self._locks[decision_type]:
            state = self._states[decision_type]
            state.boundary = clamped
            state.update_count += 1

    def freeze(self, decision_type: DecisionType) -> None:
        """Freeze expansion for a decision class (safety override)."""
        with self._locks[decision_type]:
            self._states[decision_type].frozen = True

    def unfreeze(self, decision_type: DecisionType) -> None:
        with self._locks[decision_type]:
            self._states[decision_type].frozen = False

    def is_frozen(self, decision_type: DecisionType) -> bool:
        with self._locks[decision_type]:
            return self._states[decision_type].frozen

    def get_state(self, decision_type: DecisionType) -> AutonomyState:
        with self._locks[decision_type]:
            return self._states[decision_type].model_copy()

    def all_states(self) -> Dict[str, dict]:
        return {dt.value: self.get_state(dt).snapshot() for dt in DecisionType}

    def enable_shadow_mode(self) -> None:
        self._shadow_mode = True

    def disable_shadow_mode(self) -> None:
        self._shadow_mode = False
