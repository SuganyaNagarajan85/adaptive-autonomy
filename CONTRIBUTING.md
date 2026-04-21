# Contributing to Autonomy Learning Loop

Thank you for your interest in contributing. This document covers how to get started, what's in scope, and how to submit changes.

---

## What's In Scope

| Type | Examples |
|---|---|
| **New scenarios** | Supply chain disruption, fraud wave, model drift |
| **New signal types** | Confidence divergence, drift score, SLA breach |
| **New metrics** | Latency-adjusted DDE, per-class containment |
| **Integrations** | Kafka consumer, Redis boundary store, Prometheus exporter |
| **Domain configs** | `config/fintech.yaml`, `config/healthcare.yaml` |
| **Bug fixes** | Correctness issues in boundary formulas, metrics, aggregation |
| **Tests** | Coverage for uncovered paths, property-based tests |

**Out of scope:** changes to the core `ΔB` formulas (§5) — these are the paper's formal contribution. Discuss in an issue first if you believe a formula change is warranted.

---

## Setup

```bash
git clone https://github.com/your-org/autonomy-learning-loop
cd autonomy-learning-loop
pip install -r requirements.txt

# Verify everything works
python -m pytest tests/ -v          # 42 tests should pass
python cli.py analyze --steps 100   # smoke test
```

---

## Development Guidelines

### Core invariant
**β > α must always hold.** The asymmetric safety guarantee (fast contraction, slow expansion) is non-negotiable. Any parameter change that violates this will be rejected.

### Adding a new scenario
1. Add a factory function in `src/simulation/scenarios.py` following the existing pattern (`flash_sale_scenario`, `degradation_scenario`, `recovery_scenario`)
2. Register it in `SCENARIO_REGISTRY` at the bottom of the file
3. Add at least one integration test in `tests/`
4. Run `python cli.py scenario <your_scenario>` and include the output table in your PR

### Adding a new signal type
1. Add the `SignalType` enum value in `src/feedback_signals/models.py`
2. Add its weight in `FeedbackAggregator` (`src/feedback_aggregator/aggregator.py`)
3. Add a factory method in `FeedbackCollector` if appropriate
4. Update `map_env_to_signals()` in `src/simulation/production_simulator.py` if it maps from environment state

### Adding a new metric
1. Add the field to `StepMetrics` or `ScenarioMetrics` (with a default value so existing tests don't break)
2. Add computation logic in `Metrics.compute()` (`src/experiments/production_metrics.py`)
3. Add the row to the comparison table in `ProductionExperimentRunner._print_comparison()`

---

## Testing

```bash
# Full suite
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_boundary_update.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

All PRs must pass the full test suite. New features should include tests.

---

## Submitting a PR

1. Fork the repo and create a branch: `git checkout -b feat/my-scenario`
2. Make your changes
3. Run `python -m pytest tests/ -v` — all 42 tests must pass
4. Run `python cli.py analyze` — must complete without error. The verdict (`ADAPTIVE_RECOMMENDED`, `ADAPTIVE_EFFICIENT_TUNING_NEEDED`, etc.) is parameter-dependent and not a pass/fail gate — but include it in your PR description so reviewers can assess any regression.
5. Open a PR with:
   - What you changed and why
   - Output table or plot showing the effect
   - Any parameter tuning decisions and their rationale

---

## Questions

Open an issue with the `question` label. Include the scenario name, parameter values, and the output you're seeing — that's usually enough context to give a useful answer quickly.
