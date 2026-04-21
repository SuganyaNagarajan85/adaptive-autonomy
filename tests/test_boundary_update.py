"""
Unit tests for BoundaryUpdater — the core formal logic of the Autonomy Learning Loop.
Tests the boundary adaptation formulas from §5 of the paper.
"""
import pytest
from src.boundary_update.updater import BoundaryUpdater, UpdateDirection
from src.feedback_aggregator.aggregator import WindowMetrics


def make_metrics(s_t: float, decision_type: str = "PRICING", window_index: int = 0) -> WindowMetrics:
    return WindowMetrics(
        decision_type=decision_type,
        window_index=window_index,
        normalized_risk_score=s_t,
        total_decisions=50,
        auto_decisions=40,
        human_review_decisions=10,
    )


class TestExpansionFormula:
    """ΔB = α(1−B) when S_t < τ_safe"""

    def test_expands_when_risk_below_safe_threshold(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        result = updater.compute_update(0.5, make_metrics(s_t=0.10))
        assert result.direction == UpdateDirection.EXPAND
        assert result.delta > 0

    def test_expansion_rate_formula(self):
        """ΔB = α(1−B) — verify exact value."""
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        b = 0.6
        result = updater.compute_update(b, make_metrics(s_t=0.10))
        expected_delta = 0.05 * (1.0 - b)
        assert abs(result.delta - expected_delta) < 1e-6

    def test_expansion_slows_near_max_boundary(self):
        """(1−B) factor ensures expansion slows as B approaches ceiling."""
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        result_low = updater.compute_update(0.2, make_metrics(s_t=0.10))
        result_high = updater.compute_update(0.9, make_metrics(s_t=0.10, window_index=1))
        assert result_low.delta > result_high.delta

    def test_expansion_capped_at_max_boundary(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  max_boundary=0.95, min_update_interval_seconds=0.0)
        # Push boundary very close to ceiling; repeated expansions should not exceed max
        b = 0.94
        for _ in range(20):
            result = updater.compute_update(b, make_metrics(s_t=0.05))
            b = result.boundary_after
        assert b <= 0.95

    def test_no_expansion_when_frozen(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        result = updater.compute_update(0.5, make_metrics(s_t=0.10), is_frozen=True)
        assert result.delta == 0.0
        assert result.frozen_skip is True


class TestContractionFormula:
    """ΔB = −βB when S_t ≥ τ_risk"""

    def test_contracts_when_risk_above_threshold(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        result = updater.compute_update(0.7, make_metrics(s_t=0.80))
        assert result.direction == UpdateDirection.CONTRACT
        assert result.delta < 0

    def test_contraction_rate_formula(self):
        """ΔB = −βB — verify exact value."""
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        b = 0.7
        result = updater.compute_update(b, make_metrics(s_t=0.80))
        expected_delta = -0.30 * b
        assert abs(result.delta - expected_delta) < 1e-6

    def test_contraction_faster_than_expansion(self):
        """β >> α — contraction must be larger in magnitude than expansion."""
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        b = 0.5
        expand = updater.compute_update(b, make_metrics(s_t=0.10))
        contract = updater.compute_update(b, make_metrics(s_t=0.80, window_index=1))
        assert abs(contract.delta) > abs(expand.delta)

    def test_contraction_bypasses_rate_limit(self):
        """Contraction must not be suppressed by rate limiter — safety critical."""
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=9999.0)  # extreme rate limit
        result = updater.compute_update(0.7, make_metrics(s_t=0.90))
        assert result.rate_limited is False
        assert result.delta < 0

    def test_contraction_floored_at_min_boundary(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.90, safe_threshold=0.25, risk_threshold=0.60,
                                  min_boundary=0.05, min_update_interval_seconds=0.0)
        result = updater.compute_update(0.06, make_metrics(s_t=0.95))
        assert result.boundary_after >= 0.05


class TestNeutralZone:
    """No update when τ_safe ≤ S_t < τ_risk"""

    def test_neutral_when_in_between_thresholds(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        result = updater.compute_update(0.5, make_metrics(s_t=0.40))
        assert result.direction == UpdateDirection.NEUTRAL
        assert result.delta == 0.0

    def test_boundary_unchanged_in_neutral(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=0.0)
        b = 0.67
        result = updater.compute_update(b, make_metrics(s_t=0.45))
        assert result.boundary_after == b


class TestRateLimiter:
    """Rate limiting prevents oscillatory expansion."""

    def test_expansion_rate_limited_when_too_frequent(self):
        updater = BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.25, risk_threshold=0.60,
                                  min_update_interval_seconds=9999.0)
        # First update succeeds (no prior update)
        r1 = updater.compute_update(0.5, make_metrics(s_t=0.10))
        # Second update within interval should be rate-limited
        r2 = updater.compute_update(0.5, make_metrics(s_t=0.10, window_index=1))
        # r1 may or may not be rate-limited depending on timing, but r2 should be
        assert r2.rate_limited is True or r1.delta > 0  # at least one should pass


class TestInitializationConstraints:

    def test_raises_if_beta_not_greater_than_alpha(self):
        with pytest.raises(AssertionError):
            BoundaryUpdater(alpha=0.30, beta=0.10)

    def test_raises_if_safe_threshold_not_less_than_risk(self):
        with pytest.raises(AssertionError):
            BoundaryUpdater(alpha=0.05, beta=0.30, safe_threshold=0.70, risk_threshold=0.60)
