"""
Autonomy Optimizer — cross-experiment parameter tuning engine.

After running all scenarios (flash_sale, degradation, recovery) plus the
baseline synthetic experiment, this module analyses the combined evidence to:

  1. Measure anomaly intensity   — how fast and how far incident rates spike
  2. Measure boundary depth      — how much contraction was needed vs achieved
  3. Measure headroom adequacy   — was B already too high when the anomaly hit?
  4. Measure re-expansion speed  — did the system recover autonomy correctly?

From these observations it derives concrete parameter recommendations:
  β  — contraction rate     (raise if containment is poor)
  α  — expansion rate       (lower if over-expansion, raise if under-expansion)
  τ_risk — risk threshold   (lower if contraction triggers too late)
  headroom_buffer           (raise if boundary is in headroom zone during anomalies)
  window_size               (lower if gradual degradation goes undetected)

The output is an OptimizationReport with:
  - Per-scenario RunInsights
  - Annotated ParameterRecommendations with reasoning
  - A cross-scenario summary
  - A ready-to-use recommended_params dict
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.experiments.insights import InsightsGenerator, RunInsights


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class AnomalyProfile:
    """
    Derived characterisation of how severe anomalies were in one scenario run.
    All values are computed from the adaptive run's step metrics.
    """
    scenario_name: str
    # Intensity: how much worse anomaly periods are relative to normal
    intensity_ratio: float      # anomaly_IR / normal_IR  (>1 = worse during anomaly)
    # Depth: how much the boundary actually contracted during anomaly periods
    boundary_depth: float       # initial_boundary - min_boundary_during_anomaly
    # Pre-event B: boundary level at the step immediately before first anomaly
    pre_event_boundary: float
    # Containment: fraction of static-run anomaly incidents that adaptive suppressed
    containment_score: float
    # How many steps were anomaly steps
    anomaly_steps: int
    # Peak incident rate across all steps
    peak_incident_rate: float
    # Re-expansion: fraction of lost autonomy recovered post-anomaly (recovery scenario)
    re_expansion_score: float = 0.0   # (final_B - initial_B) / (1 - initial_B); 0 if N/A
    # Whether this scenario has pathological intensity_ratio (normal_IR ≈ 0)
    pathological_intensity: bool = False


@dataclass
class ParameterRecommendation:
    """A single recommended parameter change with evidence-based reasoning."""
    parameter: str
    current_value: float
    recommended_value: float
    direction: str          # "INCREASE" | "DECREASE" | "KEEP"
    confidence: str         # "HIGH" | "MEDIUM" | "LOW"
    reasoning: str          # plain-language explanation of why
    evidence: List[str]     # specific metric values that drove this recommendation


@dataclass
class OptimizationReport:
    """Full output of the cross-experiment optimizer."""
    # Current parameters used across all runs
    current_params: Dict[str, Any]
    # Per-scenario anomaly profiles
    anomaly_profiles: Dict[str, AnomalyProfile]
    # Per-scenario RunInsights (from InsightsGenerator)
    scenario_insights: Dict[str, RunInsights]
    # Derived parameter recommendations
    recommendations: List[ParameterRecommendation]
    # Cross-scenario observations
    cross_scenario_findings: List[str]
    # Final recommended parameter set (ready to use)
    recommended_params: Dict[str, Any]
    # Overall assessment
    overall_verdict: str
    overall_summary: str


# ─── Optimizer ───────────────────────────────────────────────────────────────

class AutonomyOptimizer:
    """
    Analyses a set of experiment results and recommends parameter adjustments.

    Usage::

        optimizer = AutonomyOptimizer(current_params={...})
        report = optimizer.analyze(
            scenario_results={"flash_sale": ..., "degradation": ..., "recovery": ...},
            experiment_result=...,   # optional baseline
        )
        optimizer.print_report(report)
    """

    def __init__(self, current_params: Optional[Dict[str, Any]] = None) -> None:
        self._params = current_params or {
            "alpha": 0.025,
            "beta": 0.38,
            "safe_threshold": 0.15,
            "risk_threshold": 0.60,
            "headroom_buffer": 0.35,
            "window_size": 35,
            "max_boundary": 0.95,
        }
        self._insights_gen = InsightsGenerator()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        scenario_results: Dict[str, Any],       # name → ScenarioExperimentResult
        experiment_result: Optional[Any] = None, # ExperimentResult (baseline)
    ) -> OptimizationReport:

        # 1. Collect per-scenario insights
        scenario_insights: Dict[str, RunInsights] = {}
        if experiment_result is not None:
            scenario_insights["baseline"] = self._insights_gen.from_experiment(experiment_result)
        for name, result in scenario_results.items():
            scenario_insights[name] = self._insights_gen.from_scenario(result)

        # 2. Build anomaly profiles from step data
        profiles: Dict[str, AnomalyProfile] = {}
        for name, result in scenario_results.items():
            profiles[name] = self._build_profile(name, result)

        # 3. Derive parameter recommendations
        recommendations = self._derive_recommendations(profiles, scenario_results)

        # 4. Cross-scenario findings
        cross_findings = self._cross_scenario_findings(profiles, scenario_results)

        # 5. Build recommended params
        recommended_params = self._apply_recommendations(self._params, recommendations)

        # 6. Overall verdict
        verdict, summary = self._overall_verdict(profiles, scenario_results)

        return OptimizationReport(
            current_params=dict(self._params),
            anomaly_profiles=profiles,
            scenario_insights=scenario_insights,
            recommendations=recommendations,
            cross_scenario_findings=cross_findings,
            recommended_params=recommended_params,
            overall_verdict=verdict,
            overall_summary=summary,
        )

    # ── Profile builder ───────────────────────────────────────────────────────

    def _build_profile(self, name: str, result: Any) -> AnomalyProfile:
        a = result.adaptive_metrics
        steps = result.adaptive_steps

        raw_normal_ir = a.mean_incident_rate_normal
        normal_ir     = raw_normal_ir if raw_normal_ir > 0 else 1e-6
        anomaly_ir    = a.mean_incident_rate_anomaly
        intensity     = anomaly_ir / normal_ir
        # Flag pathological ratios: normal_IR ≈ 0 makes intensity_ratio meaningless
        pathological  = raw_normal_ir < 1e-4 and intensity > 1000

        # Find min boundary during anomaly steps
        anomaly_step_objs = [s for s in steps if s.is_anomaly]
        if anomaly_step_objs:
            min_b_during_anomaly = min(s.mean_boundary for s in anomaly_step_objs)
        else:
            min_b_during_anomaly = a.initial_mean_boundary

        boundary_depth = max(0.0, a.initial_mean_boundary - min_b_during_anomaly)

        # Pre-event boundary: mean boundary of the 5 steps before first anomaly step
        first_anomaly_idx = next(
            (i for i, s in enumerate(steps) if s.is_anomaly), None
        )
        if first_anomaly_idx and first_anomaly_idx > 0:
            pre_window = steps[max(0, first_anomaly_idx - 5): first_anomaly_idx]
            pre_event_b = sum(s.mean_boundary for s in pre_window) / len(pre_window)
        else:
            pre_event_b = a.initial_mean_boundary

        peak_ir = max((s.incident_rate for s in steps), default=0.0)

        # Re-expansion score: relevant for recovery scenario (starts degraded, should re-expand)
        re_expansion = 0.0
        if a.initial_mean_boundary < 0.40:
            denominator = max(1.0 - a.initial_mean_boundary, 1e-6)
            re_expansion = max(0.0, (a.final_mean_boundary - a.initial_mean_boundary) / denominator)

        return AnomalyProfile(
            scenario_name=name,
            intensity_ratio=round(intensity, 2),
            boundary_depth=round(boundary_depth, 3),
            pre_event_boundary=round(pre_event_b, 3),
            containment_score=round(a.containment_score, 3),
            anomaly_steps=a.anomaly_steps,
            peak_incident_rate=round(peak_ir, 4),
            re_expansion_score=round(re_expansion, 3),
            pathological_intensity=pathological,
        )

    # ── Recommendation engine ─────────────────────────────────────────────────

    def _derive_recommendations(
        self,
        profiles: Dict[str, AnomalyProfile],
        results: Dict[str, Any],
    ) -> List[ParameterRecommendation]:
        recs: List[ParameterRecommendation] = []

        alpha        = self._params["alpha"]
        beta         = self._params["beta"]
        risk_thresh  = self._params["risk_threshold"]
        safe_thresh  = self._params["safe_threshold"]
        headroom     = self._params["headroom_buffer"]
        window_size  = self._params["window_size"]
        max_b        = self._params["max_boundary"]
        soft_ceiling = max_b - headroom

        # ── β: contraction rate ─────────────────────────────────────────────
        # Exclude pathological intensity scenarios (normal_IR ≈ 0) from containment logic —
        # their containment scores are undefined since the baseline is near-zero.
        scoreable = {n: p for n, p in profiles.items() if not p.pathological_intensity}

        poor_containment = [
            (n, p) for n, p in scoreable.items()
            if p.containment_score < 0.20 and p.anomaly_steps > 0
        ]
        good_containment = [
            (n, p) for n, p in scoreable.items()
            if p.containment_score >= 0.35 and p.anomaly_steps > 0
        ]
        high_intensity = [
            (n, p) for n, p in scoreable.items()
            if p.intensity_ratio > 2.5
        ]

        if poor_containment and high_intensity:
            new_beta = min(0.65, round(beta * 1.40, 2))
            recs.append(ParameterRecommendation(
                parameter="beta",
                current_value=beta,
                recommended_value=new_beta,
                direction="INCREASE",
                confidence="HIGH",
                reasoning=(
                    f"Anomaly containment is weak (scores: "
                    f"{', '.join(f'{n}={p.containment_score:.2f}' for n, p in poor_containment)}) "
                    f"despite high anomaly intensity "
                    f"({', '.join(f'{n}={p.intensity_ratio:.1f}×' for n, p in high_intensity)}). "
                    f"Increasing β from {beta} → {new_beta} makes contraction "
                    f"{new_beta/alpha:.0f}× faster than expansion, giving the system "
                    f"more force to pull B down when risk signals arrive."
                ),
                evidence=[
                    f"{n}: containment={p.containment_score:.2f}, intensity={p.intensity_ratio:.1f}×"
                    for n, p in poor_containment
                ],
            ))
        elif poor_containment:
            new_beta = min(0.55, round(beta * 1.25, 2))
            recs.append(ParameterRecommendation(
                parameter="beta",
                current_value=beta,
                recommended_value=new_beta,
                direction="INCREASE",
                confidence="MEDIUM",
                reasoning=(
                    f"Containment below threshold in {len(poor_containment)} scenario(s). "
                    f"Modest β increase {beta} → {new_beta} improves contraction speed "
                    f"without over-correcting."
                ),
                evidence=[
                    f"{n}: containment={p.containment_score:.2f}" for n, p in poor_containment
                ],
            ))
        elif len(good_containment) == len(profiles) and profiles:
            recs.append(ParameterRecommendation(
                parameter="beta",
                current_value=beta,
                recommended_value=beta,
                direction="KEEP",
                confidence="MEDIUM",
                reasoning=(
                    f"Containment is adequate across all scenarios "
                    f"(scores ≥ 0.35). β = {beta} is working."
                ),
                evidence=[
                    f"{n}: containment={p.containment_score:.2f}" for n, p in good_containment
                ],
            ))

        # ── α: expansion rate ───────────────────────────────────────────────
        over_expanded = [
            (n, p) for n, p in profiles.items()
            if p.pre_event_boundary > soft_ceiling
        ]
        recovery_result = results.get("recovery")
        under_expanded = False
        if recovery_result:
            r_adap = recovery_result.adaptive_metrics
            b_regained = r_adap.final_mean_boundary - r_adap.initial_mean_boundary
            under_expanded = b_regained < 0.15

        if over_expanded and not under_expanded:
            new_alpha = max(0.01, round(alpha * 0.70, 3))
            recs.append(ParameterRecommendation(
                parameter="alpha",
                current_value=alpha,
                recommended_value=new_alpha,
                direction="DECREASE",
                confidence="HIGH",
                reasoning=(
                    f"Boundary drifted into headroom zone before anomalies in "
                    f"{len(over_expanded)} scenario(s) "
                    f"({', '.join(f'{n}: pre-event B={p.pre_event_boundary:.2f}' for n, p in over_expanded)}). "
                    f"Slowing expansion α {alpha} → {new_alpha} keeps the pre-event boundary "
                    f"lower, so contraction has more room to work with."
                ),
                evidence=[
                    f"{n}: pre-event B={p.pre_event_boundary:.2f} vs soft ceiling {soft_ceiling:.2f}"
                    for n, p in over_expanded
                ],
            ))
        elif under_expanded:
            new_alpha = min(0.10, round(alpha * 1.30, 3))
            confidence = "HIGH" if not over_expanded else "LOW"
            recs.append(ParameterRecommendation(
                parameter="alpha",
                current_value=alpha,
                recommended_value=new_alpha,
                direction="INCREASE",
                confidence=confidence,
                reasoning=(
                    f"Recovery scenario shows poor re-expansion (B regained: {b_regained:.2f}). "
                    f"Increasing α {alpha} → {new_alpha} allows the system to rebuild autonomy "
                    f"faster after an incident clears."
                ),
                evidence=[
                    f"recovery: initial_B={r_adap.initial_mean_boundary:.2f}, "
                    f"final_B={r_adap.final_mean_boundary:.2f}, regained={b_regained:.2f}"
                ],
            ))
        else:
            recs.append(ParameterRecommendation(
                parameter="alpha",
                current_value=alpha,
                recommended_value=alpha,
                direction="KEEP",
                confidence="MEDIUM",
                reasoning=(
                    f"Expansion rate is balanced — no over-expansion into headroom zone and "
                    f"recovery re-expansion is adequate. α = {alpha} is appropriate."
                ),
                evidence=[],
            ))

        # ── τ_risk: risk threshold ──────────────────────────────────────────
        # Only use scoreable (non-pathological) profiles here too
        late_contraction = [
            (n, p) for n, p in scoreable.items()
            if p.containment_score < 0.15 and p.intensity_ratio > 2.0 and p.anomaly_steps > 0
        ]
        if late_contraction:
            new_thresh = max(0.35, round(risk_thresh - 0.08, 2))
            recs.append(ParameterRecommendation(
                parameter="risk_threshold",
                current_value=risk_thresh,
                recommended_value=new_thresh,
                direction="DECREASE",
                confidence="HIGH",
                reasoning=(
                    f"High-intensity anomalies are not triggering contraction fast enough "
                    f"(containment < 0.15 in {len(late_contraction)} scenario(s)). "
                    f"Lowering τ_risk from {risk_thresh} → {new_thresh} means the system "
                    f"begins contracting at a lower risk score, giving it a head start on "
                    f"high-velocity events."
                ),
                evidence=[
                    f"{n}: containment={p.containment_score:.2f}, intensity={p.intensity_ratio:.1f}×"
                    for n, p in late_contraction
                ],
            ))
        else:
            recs.append(ParameterRecommendation(
                parameter="risk_threshold",
                current_value=risk_thresh,
                recommended_value=risk_thresh,
                direction="KEEP",
                confidence="MEDIUM",
                reasoning=(
                    f"Contraction triggers are firing adequately. "
                    f"τ_risk = {risk_thresh} does not need adjustment."
                ),
                evidence=[],
            ))

        # ── recovery re-expansion ──────────────────────────────────────────
        # Pathological scenarios (recovery): score on whether B re-expanded correctly
        # after the anomaly cleared, not on containment (which is meaningless).
        pathological_profiles = [(n, p) for n, p in profiles.items() if p.pathological_intensity]
        for n, p in pathological_profiles:
            if p.re_expansion_score < 0.20:
                # System isn't recovering autonomy after stabilisation
                new_alpha_rec = min(0.10, round(alpha * 1.30, 3))
                recs.append(ParameterRecommendation(
                    parameter="alpha",
                    current_value=alpha,
                    recommended_value=new_alpha_rec,
                    direction="INCREASE",
                    confidence="MEDIUM",
                    reasoning=(
                        f"Recovery scenario ({n}): system failed to re-expand autonomy after "
                        f"conditions stabilised (re_expansion_score={p.re_expansion_score:.2f}). "
                        f"Increasing α {alpha} → {new_alpha_rec} helps the system rebuild "
                        f"autonomy faster once risk signals subside."
                    ),
                    evidence=[f"{n}: re_expansion_score={p.re_expansion_score:.2f}"],
                ))
            else:
                recs.append(ParameterRecommendation(
                    parameter="alpha",
                    current_value=alpha,
                    recommended_value=alpha,
                    direction="KEEP",
                    confidence="MEDIUM",
                    reasoning=(
                        f"Recovery scenario ({n}): system correctly re-expanded autonomy "
                        f"after stabilisation (re_expansion_score={p.re_expansion_score:.2f}). "
                        f"α = {alpha} is adequate for post-incident recovery."
                    ),
                    evidence=[f"{n}: re_expansion_score={p.re_expansion_score:.2f}"],
                ))

        # ── headroom_buffer ─────────────────────────────────────────────────
        headroom_violations = [
            (n, p) for n, p in profiles.items()
            if p.pre_event_boundary > soft_ceiling and p.anomaly_steps > 0
        ]
        if headroom_violations:
            new_headroom = min(0.35, round(headroom + 0.08, 2))
            recs.append(ParameterRecommendation(
                parameter="headroom_buffer",
                current_value=headroom,
                recommended_value=new_headroom,
                direction="INCREASE",
                confidence="HIGH",
                reasoning=(
                    f"In {len(headroom_violations)} scenario(s), the boundary was already inside "
                    f"the headroom zone when the anomaly began "
                    f"({', '.join(f'{n}: B={p.pre_event_boundary:.2f}' for n, p in headroom_violations)}). "
                    f"Increasing headroom_buffer from {headroom} → {new_headroom} "
                    f"shifts the soft ceiling from {soft_ceiling:.2f} → {max_b - new_headroom:.2f}, "
                    f"ensuring the boundary enters events with more contraction room available."
                ),
                evidence=[
                    f"{n}: pre-event B={p.pre_event_boundary:.2f}, soft ceiling={soft_ceiling:.2f}"
                    for n, p in headroom_violations
                ],
            ))
        else:
            recs.append(ParameterRecommendation(
                parameter="headroom_buffer",
                current_value=headroom,
                recommended_value=headroom,
                direction="KEEP",
                confidence="MEDIUM",
                reasoning=(
                    f"Boundary was below the soft ceiling (max_boundary − headroom = {soft_ceiling:.2f}) "
                    f"in all scenarios when anomalies began. Headroom is adequate."
                ),
                evidence=[],
            ))

        # ── window_size: gradual degradation detection ──────────────────────
        degradation_result = results.get("degradation")
        if degradation_result:
            d_adap = degradation_result.adaptive_metrics
            if d_adap.containment_score < 0.10 and d_adap.anomaly_steps > 0:
                new_window = max(20, int(window_size * 0.70))
                recs.append(ParameterRecommendation(
                    parameter="window_size",
                    current_value=float(window_size),
                    recommended_value=float(new_window),
                    direction="DECREASE",
                    confidence="MEDIUM",
                    reasoning=(
                        f"Gradual degradation is going largely undetected "
                        f"(containment={d_adap.containment_score:.2f}). "
                        f"A smaller window ({window_size} → {new_window} decisions) means "
                        f"the aggregator evaluates risk more frequently, catching the slow "
                        f"error climb earlier before it becomes a hard incident."
                    ),
                    evidence=[
                        f"degradation: containment={d_adap.containment_score:.2f}, "
                        f"anomaly_IR={d_adap.mean_incident_rate_anomaly:.4f}"
                    ],
                ))
            else:
                recs.append(ParameterRecommendation(
                    parameter="window_size",
                    current_value=float(window_size),
                    recommended_value=float(window_size),
                    direction="KEEP",
                    confidence="MEDIUM",
                    reasoning=(
                        f"Degradation scenario shows adequate signal detection "
                        f"(containment={d_adap.containment_score:.2f}). "
                        f"window_size = {window_size} is appropriate."
                    ),
                    evidence=[],
                ))

        return recs

    # ── Cross-scenario findings ───────────────────────────────────────────────

    def _cross_scenario_findings(
        self,
        profiles: Dict[str, AnomalyProfile],
        results: Dict[str, Any],
    ) -> List[str]:
        findings = []
        soft_ceiling = self._params["max_boundary"] - self._params["headroom_buffer"]

        # Which scenarios did adaptive win on incident rate?
        wins  = [n for n, r in results.items() if r.incident_reduction_pct > 5]
        mixed = [n for n, r in results.items() if -5 <= r.incident_reduction_pct <= 5]
        loses = [n for n, r in results.items() if r.incident_reduction_pct < -5]

        if wins:
            findings.append(
                f"Adaptive outperforms static (>5% incident reduction) in: {', '.join(wins)}. "
                "These are conditions where the feedback loop delivers measurable value."
            )
        if mixed:
            findings.append(
                f"Marginal difference in: {', '.join(mixed)}. "
                "Adaptive achieves more automation but without clear safety improvement — "
                "parameter tuning may unlock the safety benefit."
            )
        if loses:
            # Separate out scenarios where DDE still shows adaptive value
            findings.append(
                f"Higher incident rate than static in: {', '.join(loses)}. "
                "This is partly a volume effect — adaptive runs at 80-93% autonomy vs static "
                "at ~45%, so more decisions are exposed to failure. Compare DDE and containment "
                "score for a fairer view of whether adaptive is adding governance value."
            )

        # Intensity ranking
        if profiles:
            ranked = sorted(profiles.items(), key=lambda x: x[1].intensity_ratio, reverse=True)
            def _intensity_label(n: str, p: AnomalyProfile) -> str:
                if p.pathological_intensity:
                    return f"{n} ({p.intensity_ratio:.1f}× — pathological: normal_IR≈0)"
                return f"{n} ({p.intensity_ratio:.1f}×)"
            findings.append(
                "Anomaly intensity ranking (IR_anomaly / IR_normal): "
                + ", ".join(_intensity_label(n, p) for n, p in ranked)
                + ". Higher ratio = more severe anomaly relative to baseline. "
                + "Pathological scenarios (normal_IR≈0) are scored on re-expansion, not containment."
            )

        # Headroom pattern
        over_headroom = [(n, p) for n, p in profiles.items() if p.pre_event_boundary > soft_ceiling]
        if over_headroom:
            findings.append(
                f"Pre-event boundary exceeded soft ceiling ({soft_ceiling:.2f}) in "
                f"{len(over_headroom)} scenario(s): "
                f"{', '.join(f'{n} (B={p.pre_event_boundary:.2f})' for n, p in over_headroom)}. "
                f"The system entered these events without adequate contraction headroom."
            )
        else:
            findings.append(
                f"Pre-event boundary stayed below soft ceiling ({soft_ceiling:.2f}) in all scenarios — "
                "headroom was sufficient going into each event."
            )

        # Depth vs intensity relationship
        deep_enough = [
            (n, p) for n, p in profiles.items()
            if p.boundary_depth > 0.10 and p.containment_score > 0.20
        ]
        if deep_enough:
            findings.append(
                f"In {', '.join(n for n, _ in deep_enough)}: boundary contracted meaningfully "
                f"(depth ≥ 0.10) AND containment was positive — contraction is translating "
                f"into actual incident suppression."
            )

        return findings

    # ── Apply recommendations ─────────────────────────────────────────────────

    def _apply_recommendations(
        self,
        current: Dict[str, Any],
        recs: List[ParameterRecommendation],
    ) -> Dict[str, Any]:
        updated = dict(current)
        for rec in recs:
            if rec.direction != "KEEP":
                updated[rec.parameter] = rec.recommended_value
        # Enforce β > α invariant after changes
        if updated.get("alpha", 0) >= updated.get("beta", 1):
            updated["beta"] = round(min(0.65, updated["alpha"] * 4.0), 2)
        return updated

    # ── Overall verdict ───────────────────────────────────────────────────────

    def _overall_verdict(
        self,
        profiles: Dict[str, AnomalyProfile],
        results: Dict[str, Any],
    ) -> Tuple[str, str]:
        # Exclude pathological scenarios (normal_IR≈0) from IR-based verdict —
        # comparing incident rates on a run that starts fully degraded is meaningless.
        scoreable = {n: r for n, r in results.items() if not profiles[n].pathological_intensity}
        scoreable_profiles = {n: p for n, p in profiles.items() if not p.pathological_intensity}

        reduction_pcts = [r.incident_reduction_pct for r in scoreable.values()]
        mean_reduction = sum(reduction_pcts) / len(reduction_pcts) if reduction_pcts else 0

        mean_containment = (
            sum(p.containment_score for p in scoreable_profiles.values()) / len(scoreable_profiles)
            if scoreable_profiles else 0
        )

        # DDE gain: fraction of all decisions handled correctly without human intervention
        dde_gains = [
            r.adaptive_metrics.mean_decision_driven_efficiency
            - r.static_metrics.mean_decision_driven_efficiency
            for r in scoreable.values()
        ]
        mean_dde_gain = sum(dde_gains) / len(dde_gains) if dde_gains else 0

        # Recovery re-expansion health check
        recovery_healthy = any(
            p.re_expansion_score >= 0.20
            for p in profiles.values()
            if p.pathological_intensity
        )

        if mean_dde_gain > 0.15 and mean_containment > 0.25:
            verdict = "ADAPTIVE_RECOMMENDED"
            summary = (
                f"Adaptive delivers {mean_dde_gain:.0%} more correct autonomous decisions "
                f"with {mean_containment:.2f} anomaly containment across scoreable scenarios. "
                "The system is correctly allocating human attention. "
                + ("Recovery re-expansion is healthy. " if recovery_healthy else "")
                + "Apply the recommended parameters to further improve containment."
            )
        elif mean_dde_gain > 0.10:
            verdict = "ADAPTIVE_EFFICIENT_TUNING_NEEDED"
            summary = (
                f"Adaptive delivers {mean_dde_gain:.0%} DDE gain over static — more decisions "
                f"handled correctly without human intervention. Anomaly containment is "
                f"{mean_containment:.2f} — parameter tuning (α↓, headroom↑) will reduce "
                "over-expansion and improve safety response during anomalies."
            )
        elif mean_reduction > 10 and mean_containment > 0.30:
            verdict = "STRONG_ADAPTIVE_ADVANTAGE"
            summary = (
                f"Adaptive system is clearly beneficial: mean incident reduction {mean_reduction:.1f}% "
                f"with containment {mean_containment:.2f} across scenarios. "
                "Current parameters are well-calibrated."
            )
        elif mean_reduction > 0 and mean_containment > 0.15:
            verdict = "ADAPTIVE_ADVANTAGE_WITH_TUNING"
            summary = (
                f"Adaptive system shows promise (mean reduction {mean_reduction:.1f}%, "
                f"containment {mean_containment:.2f}) but parameter adjustments would "
                "improve both safety and throughput."
            )
        elif mean_reduction >= -5:
            verdict = "REQUIRES_TUNING"
            summary = (
                f"Mixed results: mean incident reduction {mean_reduction:.1f}% on scoreable scenarios. "
                f"DDE gain is {mean_dde_gain:.0%} — adaptive is adding automation value but "
                "parameters need adjustment to improve anomaly safety."
            )
        else:
            verdict = "STATIC_PREFERRED_UNTIL_TUNED"
            summary = (
                f"Scoreable scenarios show adaptive is {abs(mean_reduction):.1f}% worse on incident rate "
                f"(volume-adjusted). DDE gain {mean_dde_gain:.0%} — the automation is working but "
                "over-expansion before anomalies is causing safety regression. "
                "Apply the recommended parameters before deploying adaptive mode."
            )

        return verdict, summary

    # ── Printer ───────────────────────────────────────────────────────────────

    def print_report(self, report: OptimizationReport) -> None:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            console = Console()
        except ImportError:
            self._plain_print(report)
            return

        # ── Anomaly profiles table ──────────────────────────────────────────
        profile_table = Table(
            title="Anomaly Profiles — Intensity & Depth Across Scenarios",
            show_header=True,
        )
        profile_table.add_column("Scenario",           style="bold cyan")
        profile_table.add_column("Intensity\n(IR ratio)",  justify="right")
        profile_table.add_column("Pre-event B",         justify="right")
        profile_table.add_column("Contraction Depth",   justify="right")
        profile_table.add_column("Containment",         justify="right")
        profile_table.add_column("Re-expansion",        justify="right")
        profile_table.add_column("Peak IR",             justify="right")

        for name, p in report.anomaly_profiles.items():
            intensity_str = f"{p.intensity_ratio:.1f}×"
            if p.pathological_intensity:
                intensity_str = f"[dim]{intensity_str}*[/dim]"
            elif p.intensity_ratio > 3:
                intensity_str = f"[red]{intensity_str}[/red]"
            elif p.intensity_ratio > 2:
                intensity_str = f"[yellow]{intensity_str}[/yellow]"
            else:
                intensity_str = f"[green]{intensity_str}[/green]"

            if p.pathological_intensity:
                containment_str = "[dim]n/a[/dim]"
            else:
                containment_str = f"{p.containment_score:.2f}"
                if p.containment_score > 0.30:
                    containment_str = f"[green]{containment_str}[/green]"
                elif p.containment_score > 0.10:
                    containment_str = f"[yellow]{containment_str}[/yellow]"
                else:
                    containment_str = f"[red]{containment_str}[/red]"

            if p.re_expansion_score > 0:
                re_exp_str = f"{p.re_expansion_score:.2f}"
                if p.re_expansion_score >= 0.20:
                    re_exp_str = f"[green]{re_exp_str}[/green]"
                else:
                    re_exp_str = f"[red]{re_exp_str}[/red]"
            else:
                re_exp_str = "[dim]—[/dim]"

            profile_table.add_row(
                name,
                intensity_str,
                f"{p.pre_event_boundary:.3f}",
                f"{p.boundary_depth:.3f}",
                containment_str,
                re_exp_str,
                f"{p.peak_incident_rate:.4f}",
            )
        console.print(profile_table)

        # ── Cross-scenario findings ─────────────────────────────────────────
        findings_lines = "\n".join(f"  • {f}" for f in report.cross_scenario_findings)
        console.print(Panel(
            findings_lines,
            title="[bold]Cross-Scenario Findings[/bold]",
            border_style="blue",
        ))

        # ── Parameter recommendations table ────────────────────────────────
        rec_table = Table(
            title="Parameter Recommendations",
            show_header=True,
        )
        rec_table.add_column("Parameter",   style="bold cyan")
        rec_table.add_column("Current",     justify="right")
        rec_table.add_column("Recommended", justify="right")
        rec_table.add_column("Direction",   justify="center")
        rec_table.add_column("Confidence",  justify="center")
        rec_table.add_column("Reasoning", no_wrap=False, max_width=55)

        dir_colors = {"INCREASE": "green", "DECREASE": "yellow", "KEEP": "white"}
        conf_colors = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "dim"}

        for rec in report.recommendations:
            dc = dir_colors.get(rec.direction, "white")
            cc = conf_colors.get(rec.confidence, "white")
            changed = rec.current_value != rec.recommended_value
            rec_table.add_row(
                rec.parameter,
                str(rec.current_value),
                f"[bold]{rec.recommended_value}[/bold]" if changed else str(rec.recommended_value),
                f"[{dc}]{rec.direction}[/{dc}]",
                f"[{cc}]{rec.confidence}[/{cc}]",
                rec.reasoning,
            )
        console.print(rec_table)

        # ── Recommended params ──────────────────────────────────────────────
        changed_params = {
            k: v for k, v in report.recommended_params.items()
            if v != report.current_params.get(k)
        }
        if changed_params:
            params_lines = "\n".join(
                f"  [cyan]{k}[/cyan]: [dim]{report.current_params.get(k)}[/dim] → [bold green]{v}[/bold green]"
                for k, v in changed_params.items()
            )
            console.print(Panel(
                params_lines,
                title="[bold]Optimized Parameter Set (changed values)[/bold]",
                border_style="green",
            ))
        else:
            console.print(Panel(
                "  No parameter changes recommended — current settings are well-calibrated.",
                title="[bold]Optimized Parameter Set[/bold]",
                border_style="green",
            ))

        # ── Overall verdict ─────────────────────────────────────────────────
        verdict_colors = {
            "STRONG_ADAPTIVE_ADVANTAGE":       "bold green",
            "ADAPTIVE_ADVANTAGE_WITH_TUNING":  "bold yellow",
            "REQUIRES_TUNING":                 "bold yellow",
            "STATIC_PREFERRED_UNTIL_TUNED":    "bold red",
        }
        vc = verdict_colors.get(report.overall_verdict, "bold white")
        console.print(Panel(
            f"[{vc}]{report.overall_verdict}[/{vc}]\n\n{report.overall_summary}",
            title="[bold]Overall Assessment[/bold]",
            border_style="cyan",
        ))

    def _plain_print(self, report: OptimizationReport) -> None:
        print(f"\nOverall: {report.overall_verdict}")
        print(report.overall_summary)
        print("\nRecommendations:")
        for rec in report.recommendations:
            if rec.direction != "KEEP":
                print(f"  {rec.parameter}: {rec.current_value} → {rec.recommended_value} ({rec.confidence})")
                print(f"    {rec.reasoning}")
