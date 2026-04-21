"""
Unit tests for FeedbackAggregator — windowed signal aggregation and S_t computation.
"""
import pytest
from src.feedback_aggregator.aggregator import FeedbackAggregator
from src.feedback_signals.models import FeedbackSignal, SignalSeverity, SignalType


def make_signal(
    sig_type: SignalType = SignalType.ROLLBACK,
    severity: SignalSeverity = SignalSeverity.MEDIUM,
    decision_type: str = "PRICING",
) -> FeedbackSignal:
    return FeedbackSignal(signal_type=sig_type, severity=severity, decision_type=decision_type)


class TestAggregation:

    def test_empty_window_returns_zero_risk(self):
        agg = FeedbackAggregator()
        agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        metrics = agg.evaluate_window()
        assert metrics["PRICING"].normalized_risk_score == 0.0

    def test_rollback_signal_increases_risk(self):
        agg = FeedbackAggregator(norm_factor=5.0)
        agg.ingest_signal(make_signal(SignalType.ROLLBACK, SignalSeverity.MEDIUM))
        agg.record_decision("PRICING", is_auto=True, outcome="ROLLBACK")
        metrics = agg.evaluate_window()
        assert metrics["PRICING"].normalized_risk_score > 0.0

    def test_multiple_signals_accumulate(self):
        agg = FeedbackAggregator(norm_factor=5.0)
        for _ in range(5):
            agg.ingest_signal(make_signal(SignalType.ROLLBACK, SignalSeverity.HIGH))
        metrics = agg.evaluate_window()
        s_t = metrics["PRICING"].normalized_risk_score
        assert s_t > 0.0

    def test_critical_severity_higher_than_low(self):
        agg1 = FeedbackAggregator(norm_factor=5.0)
        agg2 = FeedbackAggregator(norm_factor=5.0)
        agg1.ingest_signal(make_signal(SignalType.INCIDENT, SignalSeverity.CRITICAL))
        agg2.ingest_signal(make_signal(SignalType.INCIDENT, SignalSeverity.LOW))
        m1 = agg1.evaluate_window()
        m2 = agg2.evaluate_window()
        assert m1["PRICING"].normalized_risk_score > m2["PRICING"].normalized_risk_score

    def test_s_t_bounded_between_0_and_1(self):
        agg = FeedbackAggregator(norm_factor=1.0)  # small norm = easy saturation
        for _ in range(100):
            agg.ingest_signal(make_signal(SignalType.INCIDENT, SignalSeverity.CRITICAL))
        metrics = agg.evaluate_window()
        s_t = metrics["PRICING"].normalized_risk_score
        assert 0.0 <= s_t <= 1.0

    def test_window_resets_after_evaluation(self):
        agg = FeedbackAggregator(norm_factor=5.0)
        agg.ingest_signal(make_signal(SignalType.ROLLBACK))
        agg.evaluate_window()  # closes window 0
        # Window 1 should start empty
        agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        metrics = agg.evaluate_window()
        assert metrics["PRICING"].normalized_risk_score == 0.0

    def test_per_class_separation(self):
        agg = FeedbackAggregator(norm_factor=5.0)
        agg.ingest_signal(make_signal(SignalType.ROLLBACK, decision_type="PRICING"))
        agg.ingest_signal(make_signal(SignalType.OVERRIDE, decision_type="FRAUD_DETECTION"))
        metrics = agg.evaluate_window()
        # Both classes should appear independently
        assert "PRICING" in metrics
        assert "FRAUD_DETECTION" in metrics


class TestDecisionCounting:

    def test_auto_and_human_counted_separately(self):
        agg = FeedbackAggregator()
        agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        agg.record_decision("PRICING", is_auto=False, outcome="SUCCESS")
        metrics = agg.evaluate_window()
        assert metrics["PRICING"].auto_decisions == 2
        assert metrics["PRICING"].human_review_decisions == 1
        assert metrics["PRICING"].total_decisions == 3

    def test_adverse_outcomes_counted(self):
        agg = FeedbackAggregator()
        agg.record_decision("PRICING", is_auto=True, outcome="ROLLBACK")
        agg.record_decision("PRICING", is_auto=True, outcome="OVERRIDE")
        agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        metrics = agg.evaluate_window()
        m = metrics["PRICING"]
        assert m.rollback_rate == pytest.approx(1 / 3, abs=1e-4)
        assert m.override_rate == pytest.approx(1 / 3, abs=1e-4)

    def test_autonomy_utilization(self):
        agg = FeedbackAggregator()
        for _ in range(8):
            agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        for _ in range(2):
            agg.record_decision("PRICING", is_auto=False, outcome="SUCCESS")
        metrics = agg.evaluate_window()
        assert metrics["PRICING"].autonomy_utilization == pytest.approx(0.8, abs=1e-4)

    def test_window_index_increments(self):
        agg = FeedbackAggregator()
        agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        m1 = agg.evaluate_window()
        agg.record_decision("PRICING", is_auto=True, outcome="SUCCESS")
        m2 = agg.evaluate_window()
        assert m1["PRICING"].window_index == 0
        assert m2["PRICING"].window_index == 1
