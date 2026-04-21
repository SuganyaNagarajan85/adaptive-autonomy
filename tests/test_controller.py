"""
Unit tests for AutonomyController — routing, boundary management, thread safety.
"""
import threading
import pytest
from src.autonomy_controller.controller import AutonomyController
from src.autonomy_controller.models import RoutingOutcome
from src.decision_engine.models import DecisionType, ExecutionMode


class TestRouting:

    def test_routes_auto_when_draw_below_boundary(self):
        """With B=1.0 every draw < B, so all decisions go AUTO."""
        controller = AutonomyController(initial_boundary=1.0, seed=42)
        for dt in DecisionType:
            controller.set_boundary(dt, 1.0)
        routing = controller.route(DecisionType.PRICING)
        assert routing.outcome == RoutingOutcome.AUTO_EXECUTE

    def test_routes_human_review_when_draw_above_boundary(self):
        """With B=0.0 every draw >= B, so all decisions go HUMAN_REVIEW."""
        controller = AutonomyController(initial_boundary=0.0, min_boundary=0.0, seed=42)
        for dt in DecisionType:
            controller.set_boundary(dt, 0.0)
        routing = controller.route(DecisionType.FRAUD_DETECTION)
        assert routing.outcome == RoutingOutcome.HUMAN_REVIEW

    def test_routing_distribution_approximates_boundary(self):
        """Over many samples, fraction auto-routed ≈ B."""
        b = 0.7
        controller = AutonomyController(initial_boundary=b, seed=99)
        auto_count = sum(
            1 for _ in range(10_000)
            if controller.route(DecisionType.PRODUCT_RECOMMENDATION).outcome
            in (RoutingOutcome.AUTO_EXECUTE, RoutingOutcome.SHADOW_EXECUTE)
        )
        auto_fraction = auto_count / 10_000
        assert abs(auto_fraction - b) < 0.03  # within 3% of expected B

    def test_shadow_mode_routes_as_shadow(self):
        controller = AutonomyController(initial_boundary=1.0, shadow_mode=True, seed=1)
        for dt in DecisionType:
            controller.set_boundary(dt, 1.0)
        routing = controller.route(DecisionType.OFFER_SELECTION)
        assert routing.outcome == RoutingOutcome.SHADOW_EXECUTE
        assert routing.execution_mode == ExecutionMode.SHADOW

    def test_execution_mode_matches_outcome(self):
        controller = AutonomyController(initial_boundary=0.5, seed=42)
        for _ in range(100):
            routing = controller.route(DecisionType.NOTIFICATION)
            if routing.outcome == RoutingOutcome.AUTO_EXECUTE:
                assert routing.execution_mode == ExecutionMode.AUTO
            elif routing.outcome == RoutingOutcome.HUMAN_REVIEW:
                assert routing.execution_mode == ExecutionMode.HUMAN_REVIEW


class TestBoundaryManagement:

    def test_set_boundary_updates_value(self):
        controller = AutonomyController(initial_boundary=0.5)
        controller.set_boundary(DecisionType.PRICING, 0.8)
        assert controller.get_boundary(DecisionType.PRICING) == pytest.approx(0.8)

    def test_set_boundary_clamped_to_max(self):
        controller = AutonomyController(max_boundary=0.90)
        controller.set_boundary(DecisionType.PRICING, 0.99)
        assert controller.get_boundary(DecisionType.PRICING) <= 0.90

    def test_set_boundary_clamped_to_min(self):
        controller = AutonomyController(min_boundary=0.05)
        controller.set_boundary(DecisionType.PRICING, 0.00)
        assert controller.get_boundary(DecisionType.PRICING) >= 0.05

    def test_class_overrides_applied(self):
        overrides = {"FRAUD_DETECTION": 0.25, "PRODUCT_RECOMMENDATION": 0.70}
        controller = AutonomyController(initial_boundary=0.5, class_overrides=overrides)
        assert controller.get_boundary(DecisionType.FRAUD_DETECTION) == pytest.approx(0.25)
        assert controller.get_boundary(DecisionType.PRODUCT_RECOMMENDATION) == pytest.approx(0.70)

    def test_non_overridden_class_uses_initial(self):
        overrides = {"FRAUD_DETECTION": 0.25}
        controller = AutonomyController(initial_boundary=0.5, class_overrides=overrides)
        assert controller.get_boundary(DecisionType.PRICING) == pytest.approx(0.5)

    def test_update_count_increments(self):
        controller = AutonomyController(initial_boundary=0.5)
        controller.set_boundary(DecisionType.PRICING, 0.6)
        controller.set_boundary(DecisionType.PRICING, 0.7)
        state = controller.get_state(DecisionType.PRICING)
        assert state.update_count == 2


class TestFreezeUnfreeze:

    def test_freeze_sets_frozen_flag(self):
        controller = AutonomyController()
        controller.freeze(DecisionType.FRAUD_DETECTION)
        assert controller.is_frozen(DecisionType.FRAUD_DETECTION) is True

    def test_unfreeze_clears_frozen_flag(self):
        controller = AutonomyController()
        controller.freeze(DecisionType.FRAUD_DETECTION)
        controller.unfreeze(DecisionType.FRAUD_DETECTION)
        assert controller.is_frozen(DecisionType.FRAUD_DETECTION) is False

    def test_freeze_does_not_affect_other_classes(self):
        controller = AutonomyController()
        controller.freeze(DecisionType.FRAUD_DETECTION)
        assert controller.is_frozen(DecisionType.PRICING) is False


class TestThreadSafety:

    def test_concurrent_routing_does_not_raise(self):
        """1000 concurrent route calls should not raise or corrupt state."""
        controller = AutonomyController(initial_boundary=0.5, seed=1)
        errors = []

        def route_many():
            try:
                for _ in range(100):
                    controller.route(DecisionType.PRODUCT_RECOMMENDATION)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=route_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_set_boundary_does_not_corrupt(self):
        controller = AutonomyController(initial_boundary=0.5)
        values = [0.3, 0.4, 0.5, 0.6, 0.7]

        def set_val(v):
            controller.set_boundary(DecisionType.PRICING, v)

        threads = [threading.Thread(target=set_val, args=(v,)) for v in values]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final boundary should be one of the valid values (no corruption)
        final = controller.get_boundary(DecisionType.PRICING)
        assert 0.05 <= final <= 0.95
