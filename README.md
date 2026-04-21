# Autonomy Learning Loop (ALL)

[![Tests](https://github.com/SuganyaNagarajan85/adaptive-autonomy/actions/workflows/tests.yml/badge.svg)](https://github.com/SuganyaNagarajan85/adaptive-autonomy/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Adaptive control layer for AI systems that dynamically adjusts automation boundaries based on real-time operational feedback — without retraining.**

---

## The Problem with Static Automation

Most production AI systems operate with a **fixed automation rate** — a static threshold that never adjusts. This breaks in two predictable ways:

| Failure Mode | What Happens | Cost |
|---|---|---|
| **Autonomy stagnation** | System stays conservative even when safe | Wasted human review capacity |
| **Runaway expansion** | No contraction when incidents spike | Flash sales, fraud waves cause cascading failures |

Static thresholds can't earn trust. They can't respond to risk. They don't learn.

**The Autonomy Learning Loop treats the automation boundary as a continuously governed property** — expanding slowly when signals are safe, contracting fast when risk grows, and jumping to conservative mode when spikes are detected.

---

## Measured Impact

From production-scenario simulations across three stress test conditions:

| Scenario | DDE Gain | Containment | Verdict |
|---|---|---|---|
| Flash Sale (5× traffic burst) | **+29% more correct autonomous decisions** | 59% of anomaly incidents contained | `ADAPTIVE_WINS` |
| Gradual Degradation (5%→45% error climb) | **+28% DDE** | Trend detected before threshold breach | `ADAPTIVE_EFFICIENT` |
| Recovery (starts degraded, stabilises) | Re-expansion score: **0.45** | B rebuilt correctly as conditions cleared | `RECOVERY_HEALTHY` |

> **Decision-Driven Efficiency (DDE):** fraction of ALL decisions handled autonomously AND correctly without human intervention. Adaptive: 71–82% vs Static: 43–48%.

Autonomous success rate: **94–96%** — statistically identical to static. The extra automation doesn't sacrifice quality.

---

## Run in 60 Seconds

```bash
git clone https://github.com/SuganyaNagarajan85/adaptive-autonomy
cd adaptive-autonomy
pip install -r requirements.txt

# Full cross-scenario analysis with optimization report
python cli.py analyze
```

**Expected output:**

```
Full Autonomy Analysis
  α=0.025  β=0.38  τ_safe=0.15  τ_risk=0.6  headroom=0.35  window_size=35

Step 2/5: Running scenario: FLASH_SALE ...
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                     ┃ Static ┃ Adaptive ┃ Δ (adaptive − static) ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ Decision-Driven Efficiency │ 0.4282 │   0.7171 │               +0.2889 │
│ Containment Score          │ 0.0000 │   0.5911 │               +0.5911 │
│ Mean Autonomy Util.        │ 0.4543 │   0.7494 │               +0.2951 │
│ IR Anomaly Period          │ 0.0265 │   0.0145 │               -0.0120 │
└────────────────────────────┴────────┴──────────┴───────────────────────┘

Verdict: ADAPTIVE_WINS
Overall: ADAPTIVE_RECOMMENDED
```

Generates plots per scenario + `outputs/analysis/optimization_report.json`.

---

## How It Works

```
                   ┌─────────────────────────────┐
                   │   PRODUCTION ENVIRONMENT     │
                   │  traffic · errors · latency  │
                   └──────────────┬──────────────┘
                                  │ operational signals
                                  ▼
                   ┌─────────────────────────────┐
                   │     AUTONOMY CONTROLLER      │
                   │   B ∈ [0,1] per decision     │
                   │   u < B → AUTO EXECUTE       │
                   │   u ≥ B → HUMAN REVIEW       │
                   └──────────────┬──────────────┘
                        │                │
               outcome feedback    human resolution
                        │
                        ▼
          ┌──────────────────────────────┐
          │    FEEDBACK AGGREGATOR        │
          │  S_t = risk score (per window)│
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │     BOUNDARY UPDATER          │
          │                               │
          │  S_t < τ_safe: ΔB = +α(1−B)  │  ← slow expansion
          │  S_t ≥ τ_risk: ΔB = −β·B     │  ← fast contraction
          │  trend↑:       ΔB = −0.35β·B │  ← soft trend contraction
          │  spike:        B  → B_floor   │  ← circuit breaker
          │                               │
          │  β >> α  (asymmetric safety)  │
          └──────────────┬───────────────┘
                         │
                    B_t+1 applied
                    + audit logged
```

**Key invariant:** β >> α — the system always contracts faster than it expands. Safety asymmetry is built into the formula, not a policy choice.

---

## Architecture

```
src/
├── autonomy_controller/      # Governance gate — probabilistic routing per class
├── boundary_update/
│   ├── updater.py            # ΔB formulas, rate limiter, headroom taper
│   └── spike_detector.py     # Circuit breaker + slow-trend detection
├── feedback_aggregator/      # Windowed S_t computation per decision type
├── feedback_signals/         # Async signal ingestion (rollback/override/incident)
├── governance/               # Audit logger (JSONL), human review queue, shadow mode
├── simulation/               # Production simulator, pattern generator, scenarios
└── experiments/              # Runner, metrics, insights, optimizer, visualizer

api/        # FastAPI — route decisions, inspect boundaries, manage review queue
ui/         # Streamlit — human-in-the-loop dashboard
cli.py      # Click CLI — run | experiment | scenario | sweep | analyze
config/     # default.yaml + ecommerce.yaml
tests/      # 42 unit tests: boundary formulas, aggregation, routing, thread safety
```

**Component flow:** Decision → AutonomyController (route) → FeedbackCollector (signal) → FeedbackAggregator (S_t) → BoundaryUpdater (ΔB) → AuditLogger

---

## Three Stress Scenarios

| Scenario | What It Tests | Signal Pattern |
|---|---|---|
| **Flash Sale** | Rapid contraction under sudden load | 5× traffic burst at step 100, 50 steps |
| **Gradual Degradation** | Slow-trend detection before threshold breach | Error rate 5% → 45% over 320 steps |
| **Recovery** | Earned re-expansion after incident clears | Starts degraded (50% errors), stabilises to 6% |

These aren't synthetic edge cases — they map directly to production failure modes: traffic spikes, memory leaks / upstream decay, and post-incident remediation.

Run any scenario:

```bash
python cli.py scenario flash_sale
python cli.py scenario degradation --steps 600
python cli.py scenario recovery --seed 99
```

---

## What Makes This Different

| Approach | Problem |
|---|---|
| Reinforcement Learning | Requires training, reward engineering, offline data |
| Static thresholds | Never adapts; stagnates or over-expands |
| Confidence-only gating | Single-signal; blind to operational outcomes |
| **Autonomy Learning Loop** | **Learns from operational signals in real-time. No training. No reward function.** |

This system does not predict risk — it **measures it from outcomes** (rollbacks, overrides, incidents) and adjusts boundaries continuously. The feedback loop is attribution-tolerant: it works even without perfect causal attribution of which decision caused which outcome.

---

## Safety Mechanisms

| Mechanism | What It Does |
|---|---|
| **Circuit breaker** | Detects S_t growing ≥ 1.5× per window for 2+ consecutive windows → immediately jumps B to floor (0.15) |
| **Trend contraction** | Linear regression over last 4 S_t windows — detects slow degradation invisible to single-window thresholds |
| **Sustained-safety lock** | Post-circuit-breaker: expansion blocked until BOTH lock window expires AND N consecutive safe windows observed |
| **Headroom buffer** | Soft ceiling below max_boundary — expansion tapers linearly to zero, reserves contraction room |
| **Rate-limited expansion** | `min_update_interval` prevents oscillatory churn on expansion |
| **Fast contraction bypass** | Contractions skip the rate limiter — safety-critical response is never throttled |
| **Human review queue** | SLA-tracked priority queue for decisions above threshold |
| **Shadow mode** | Run adaptive alongside production without acting — compare outcomes before committing |
| **Audit log** | Append-only JSONL — every boundary change, override, and circuit breaker event |

---

## Real-World Applicability

This architecture is domain-agnostic. The boundary `B`, feedback signals, and update formulas apply wherever automated decisions need governed autonomy:

| Domain | Autonomous Decisions | Risk Signals |
|---|---|---|
| **E-commerce** | Recommendations, pricing, fraud detection | Rollbacks, overrides, incident rate |
| **AIOps / SRE** | Auto-remediation, scaling, failover | Error rates, latency, rollback rate |
| **Fraud & Trust** | Transaction approval, account actions | False positive rate, dispute rate |
| **AI Agents** | Tool calls, web actions, code execution | Human corrections, task failures |
| **Supply Chain** | Routing, inventory, order fulfilment | Delay rate, exception rate |

Map your outcome signals to `SignalType` — the rest of the loop is unchanged.

---

## Quickstart (Programmatic)

```python
from src.simulation.scenarios import flash_sale_scenario
from src.experiments.production_runner import ProductionExperimentRunner

runner = ProductionExperimentRunner(output_dir="outputs")
result = runner.run_scenario(flash_sale_scenario(total_steps=400))

result.print_comparison()
print(f"DDE gain: {result.adaptive_metrics.mean_decision_driven_efficiency:.1%}")
print(f"Containment: {result.adaptive_metrics.containment_score:.2f}")
```

```python
# Plug in your own scenario
from src.simulation.scenarios import ScenarioConfig

scenario = ScenarioConfig(
    name="my_service",
    description="Custom API gateway scenario",
    alpha=0.025,
    beta=0.38,
    safe_threshold=0.15,
    risk_threshold=0.60,
    headroom_buffer=0.35,
    total_steps=500,
)
result = runner.run_scenario(scenario)
```

---

## Installation

```bash
# Python 3.9+
pip install -r requirements.txt

# Run all 42 tests
python -m pytest tests/ -v

# Start REST API
uvicorn api.main:app --reload --port 8000
# → Swagger UI: http://localhost:8000/docs

# Start human-in-the-loop dashboard
streamlit run ui/dashboard.py
```

---

## Generated Outputs

Each scenario run produces:

| Plot | What It Shows |
|---|---|
| `autonomy.png` | B over time: static (flat) vs adaptive (evolving), with anomaly shading + CB markers |
| `incidents.png` | Per-step incident rate + cumulative incident count comparison |
| `efficiency.png` | Efficiency bars: overall / normal period / anomaly period |
| `system_behavior.png` | Three-panel: traffic · error rate · autonomy boundary on shared time axis |

Full analysis also produces `outputs/analysis/optimization_report.json` with per-scenario anomaly profiles, evidence-based parameter recommendations, and overall verdict.

---

## Configuration

```yaml
# config/default.yaml
boundary_update:
  alpha: 0.025          # expansion rate  — lower = more conservative growth
  beta: 0.38            # contraction rate — higher = faster safety response
  safe_threshold: 0.15  # S_t below this  → expand (also gates trend detector)
  risk_threshold: 0.60  # S_t above this  → contract
  headroom_buffer: 0.35 # reserved gap below max_boundary
  window_size: 35       # decisions per evaluation window

  # Circuit breaker + trend detection
  spike_growth_threshold: 1.5      # S_t growth ratio to count as spike
  spike_consecutive_windows: 2     # windows before CB fires
  spike_conservative_boundary: 0.15
  spike_recovery_lock_windows: 10
  sustained_safe_windows: 3        # consecutive safe windows to exit CB lock
  spike_trend_window_count: 4      # history depth for slope computation
  spike_trend_slope_threshold: 0.02
```

**Rule**: β >> α always. A ratio of 10×–15× is a safe production starting point.

---

## Extending the System

- **Custom signals**: add a `SignalType` and emit via `FeedbackCollector.emit()`
- **Persistent boundaries**: swap `AutonomyController._states` with Redis/Postgres
- **Real ML inference**: replace `DecisionEngine.make_decision()` with your model
- **Kafka integration**: replace `FeedbackCollector` with a Kafka consumer implementation
- **Custom scenarios**: call `ScenarioConfig(...)` directly or add a factory to `scenarios.py`
- **Pattern replay**: call `ProductionSimulator.full_series()` to export and replay recorded signals

---

## Suggested Repo Names

| Name | Positioning |
|---|---|
| `adaptive-autonomy` | Clear, searchable, domain-agnostic |
| `autonomy-controller` | Emphasises the control-plane role |
| `safe-autonomy-loop` | Leads with safety — relevant for AI governance audiences |

---

## Research Foundation

Implements the formal boundary adaptation system from:

> *"Feedback-Driven Autonomy: Learning Safe Automation Boundaries in Large-Scale Decision Systems"*

Core formulas (§5):

```
Expansion:  ΔB_t = α(1 − B_t)   if S_t < τ_safe
Contract:   ΔB_t = −β · B_t      if S_t ≥ τ_risk
Neutral:    ΔB_t = 0              otherwise

β >> α  — asymmetric safety guarantee
B_t ∈ [B_min, B_max]  — bounded by design
```

Extended with production-validated additions: spike circuit breaker, cross-window trend detection, sustained-safety requirement, and headroom buffer — all validated across three canonical scenario types.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — new scenarios, signal types, domain configs, and integrations are all welcome.

---

## License

MIT
