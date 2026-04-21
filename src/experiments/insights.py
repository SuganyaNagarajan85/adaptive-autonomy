"""
Insights Generator — produces plain-language findings and takeaways from
experiment comparison results.

Works with both:
  - ExperimentResult   (from ExperimentRunner, synthetic anomaly injection)
  - ScenarioExperimentResult (from ProductionExperimentRunner, production scenarios)

Each result is analysed across four dimensions:
  1. Safety    — incident / override / rollback rates
  2. Autonomy  — how much automation was achieved and at what cost
  3. Boundary  — how the learning boundary behaved (expansion, contraction, stability)
  4. Anomaly   — how well the system handled stress periods specifically
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

SCENARIO_CONTEXT = {
    "flash_sale": (
        "Flash sale tests rapid contraction under sudden load — the system must "
        "detect a spike via ROLLBACK/INCIDENT signals and pull B down fast enough "
        "to route high-risk decisions to humans before they cause incidents."
    ),
    "degradation": (
        "Degradation tests early-warning detection — errors climb slowly, so the "
        "system must pick up the trend through OVERRIDE and latency INCIDENT signals "
        "before the error rate peaks. Boundary should start contracting well before "
        "the worst window."
    ),
    "recovery": (
        "Recovery tests earned re-expansion — the system starts degraded (B low) "
        "and should re-expand autonomy only once sustained safe windows are observed. "
        "Too-fast re-expansion risks a relapse; too-slow re-expansion wastes the "
        "operational headroom the fix has created."
    ),
}


@dataclass
class RunInsights:
    """Structured insights from one static-vs-adaptive comparison."""
    verdict: str                      # ADAPTIVE_WINS | STATIC_WINS | MIXED | INCONCLUSIVE
    verdict_reason: str               # one-line summary
    findings: List[str] = field(default_factory=list)   # what the numbers show
    takeaways: List[str] = field(default_factory=list)  # what it means / what to tune
    warnings: List[str] = field(default_factory=list)   # red flags


class InsightsGenerator:
    """
    Derives RunInsights from experiment result objects.

    Usage::

        gen = InsightsGenerator()
        insights = gen.from_experiment(result)        # ExperimentResult
        insights = gen.from_scenario(result)          # ScenarioExperimentResult
        gen.print(insights)
    """

    # ── Public entry points ───────────────────────────────────────────────────

    def from_experiment(self, result) -> RunInsights:
        """Analyse an ExperimentResult (synthetic anomaly injection)."""
        s = result.static_summary
        a = result.adaptive_summary
        ir_delta = result.incident_reduction_pct      # positive = adaptive better
        or_delta = result.override_reduction_pct

        verdict, verdict_reason = self._verdict_experiment(ir_delta, a, s)
        findings  = self._findings_experiment(ir_delta, or_delta, a, s)
        takeaways = self._takeaways_experiment(ir_delta, a, s)
        warnings  = self._warnings_experiment(ir_delta, a, s)

        return RunInsights(
            verdict=verdict,
            verdict_reason=verdict_reason,
            findings=findings,
            takeaways=takeaways,
            warnings=warnings,
        )

    def from_scenario(self, result) -> RunInsights:
        """Analyse a ScenarioExperimentResult (production scenario)."""
        s = result.static_metrics
        a = result.adaptive_metrics
        ir_delta = result.incident_reduction_pct
        eff_gain = result.efficiency_gain
        scenario_name = result.scenario.name

        verdict, verdict_reason = self._verdict_scenario(ir_delta, eff_gain, a, s, scenario_name)
        findings  = self._findings_scenario(ir_delta, eff_gain, a, s, scenario_name)
        takeaways = self._takeaways_scenario(ir_delta, eff_gain, a, s, scenario_name)
        warnings  = self._warnings_scenario(ir_delta, a, s, scenario_name)

        return RunInsights(
            verdict=verdict,
            verdict_reason=verdict_reason,
            findings=findings,
            takeaways=takeaways,
            warnings=warnings,
        )

    # ── Verdict logic ─────────────────────────────────────────────────────────

    def _verdict_experiment(self, ir_delta, a, s) -> tuple[str, str]:
        if ir_delta > 10:
            return "ADAPTIVE_WINS", (
                f"Adaptive reduced incidents by {ir_delta:.1f}% vs static — "
                "feedback loop is delivering measurable safety improvement."
            )
        if ir_delta > 0:
            return "ADAPTIVE_WINS", (
                f"Adaptive reduced incidents by {ir_delta:.1f}% with "
                f"{a.mean_autonomy_utilization:.0%} autonomy utilization."
            )
        if ir_delta < -10:
            return "STATIC_WINS", (
                f"Adaptive caused {abs(ir_delta):.1f}% MORE incidents than static — "
                "the boundary over-expanded and exposed too many decisions to automation."
            )
        if ir_delta < 0:
            return "MIXED", (
                f"Adaptive produced marginally more incidents ({abs(ir_delta):.1f}%) "
                f"but achieved {(a.mean_autonomy_utilization - s.mean_autonomy_utilization):.0%} "
                "more automation throughput."
            )
        return "INCONCLUSIVE", "Incident rates are nearly identical across both systems."

    def _verdict_scenario(self, ir_delta, eff_gain, a, s, scenario_name: str = "") -> tuple[str, str]:
        dde_gain = a.mean_decision_driven_efficiency - s.mean_decision_driven_efficiency
        autonomy_gap = a.mean_autonomy_utilization - s.mean_autonomy_utilization

        # ── Recovery: score on re-expansion, not incident rate ────────────────
        # The whole run is anomaly; incident rate comparison is meaningless.
        # What matters: did the system correctly earn back autonomy as conditions stabilised?
        if scenario_name == "recovery":
            re_expansion = 0.0
            if a.initial_mean_boundary < 0.40:
                re_expansion = max(
                    0.0,
                    (a.final_mean_boundary - a.initial_mean_boundary)
                    / max(1.0 - a.initial_mean_boundary, 1e-6),
                )
            if re_expansion >= 0.30 and a.circuit_breaker_activations < 15:
                return "RECOVERY_HEALTHY", (
                    f"Re-expansion score {re_expansion:.2f} — system correctly rebuilt autonomy "
                    f"after stabilisation. CB fired {a.circuit_breaker_activations}× (controlled)."
                )
            elif re_expansion >= 0.15:
                return "RECOVERY_HEALTHY", (
                    f"Re-expansion score {re_expansion:.2f} — boundary expanding as conditions "
                    f"stabilised. CB fired {a.circuit_breaker_activations}× — watch for oscillation."
                )
            else:
                return "RECOVERY_OSCILLATING", (
                    f"Re-expansion score {re_expansion:.2f} — system struggled to rebuild autonomy. "
                    f"CB fired {a.circuit_breaker_activations}× — oscillation likely suppressing "
                    "re-expansion."
                )

        # ── All other scenarios: DDE + containment first, IR second ──────────
        # Volume effect: adaptive at 80-93% autonomy will always generate more total
        # incidents than static at 45-50%. That's exposure, not quality regression.
        # The right question: are the EXTRA autonomous decisions succeeding?

        if dde_gain > 0.15 and a.containment_score > 0.25:
            return "ADAPTIVE_WINS", (
                f"Adaptive handles {dde_gain:+.1%} more decisions correctly without human "
                f"intervention and contains {a.containment_score:.0%} of anomaly incidents — "
                "better automation AND better safety during stress."
            )
        if dde_gain > 0.10:
            return "ADAPTIVE_EFFICIENT", (
                f"Adaptive handles {dde_gain:+.1%} more decisions correctly without human "
                f"intervention ({autonomy_gap:.0%} more automation). Higher total IR reflects "
                "volume effect — same per-decision quality, more decisions made."
            )
        if a.containment_score > 0.30 and ir_delta < 0:
            return "ADAPTIVE_EFFICIENT", (
                f"Adaptive contained {a.containment_score:.0%} of anomaly incidents despite "
                f"higher total IR (volume effect from {autonomy_gap:.0%} more automation). "
                "Safety during stress is better; tune α/headroom to reduce pre-event over-expansion."
            )
        if ir_delta > 5 and eff_gain > 0:
            return "ADAPTIVE_WINS", (
                f"Adaptive reduced incidents by {ir_delta:.1f}% and gained "
                f"{eff_gain:+.3f} efficiency — better safety AND better throughput."
            )
        if dde_gain < 0.05 and ir_delta < -20:
            return "STATIC_WINS", (
                f"Adaptive shows minimal DDE gain ({dde_gain:+.1%}) while producing "
                f"{abs(ir_delta):.1f}% more incidents — boundary expansion is not delivering "
                "automation value. Review α and headroom_buffer."
            )
        if dde_gain > 0.05:
            return "MIXED", (
                f"Moderate DDE gain ({dde_gain:+.1%}) with {abs(ir_delta):.1f}% more total "
                f"incidents (volume effect). Parameter tuning should improve the balance."
            )
        return "INCONCLUSIVE", (
            "Differences are small. Consider tuning α/β or running longer scenarios."
        )

    # ── Findings (what the numbers show) ─────────────────────────────────────

    def _findings_experiment(self, ir_delta, or_delta, a, s) -> List[str]:
        findings = []

        # Incident rate
        if ir_delta > 0:
            findings.append(
                f"Incident rate: adaptive {a.mean_incident_rate:.4f} vs static {s.mean_incident_rate:.4f} "
                f"({ir_delta:+.1f}% reduction)."
            )
        else:
            findings.append(
                f"Incident rate: adaptive {a.mean_incident_rate:.4f} vs static {s.mean_incident_rate:.4f} "
                f"({ir_delta:+.1f}% — adaptive is worse)."
            )

        # Override rate
        findings.append(
            f"Override rate: adaptive {a.mean_override_rate:.4f} vs static {s.mean_override_rate:.4f} "
            f"({or_delta:+.1f}%)."
        )

        # Autonomy utilization
        au_diff = (a.mean_autonomy_utilization - s.mean_autonomy_utilization) * 100
        findings.append(
            f"Autonomy utilization: adaptive {a.mean_autonomy_utilization:.1%} vs static "
            f"{s.mean_autonomy_utilization:.1%} ({au_diff:+.1f}pp)."
        )

        # Boundary movement
        b_start = 0.5  # initial boundary is always the configured value
        b_end   = a.final_mean_boundary
        if a.boundary_stability > 0.05:
            direction = "expanded to" if b_end > b_start else "contracted to"
            findings.append(
                f"Boundary was active (σ={a.boundary_stability:.3f}): {direction} {b_end:.2f} "
                "over the run — the feedback loop was responding to signals."
            )
        else:
            findings.append(
                f"Boundary barely moved (σ={a.boundary_stability:.3f}), ending at {b_end:.2f} — "
                "few update signals reached the governor."
            )

        # Anomaly resilience
        res_diff = a.anomaly_resilience_score - s.anomaly_resilience_score
        if res_diff > 0.02:
            findings.append(
                f"Anomaly resilience: adaptive {a.anomaly_resilience_score:.3f} vs "
                f"static {s.anomaly_resilience_score:.3f} — adaptive handled stress periods better."
            )
        elif res_diff < -0.02:
            findings.append(
                f"Anomaly resilience: adaptive {a.anomaly_resilience_score:.3f} vs "
                f"static {s.anomaly_resilience_score:.3f} — adaptive was more vulnerable to anomalies."
            )

        return findings

    def _findings_scenario(self, ir_delta, eff_gain, a, s, scenario_name) -> List[str]:
        findings = []

        dde_gain = a.mean_decision_driven_efficiency - s.mean_decision_driven_efficiency
        autonomy_gap = a.mean_autonomy_utilization - s.mean_autonomy_utilization

        # Incident comparison — with volume effect note
        sign = "↓" if ir_delta > 0 else "↑"
        findings.append(
            f"Incident rate: adaptive {a.mean_incident_rate:.4f} vs static {s.mean_incident_rate:.4f} "
            f"({sign}{abs(ir_delta):.1f}%)."
        )
        # Explain volume effect when IR is higher but autonomy gap is large
        if ir_delta < 0 and autonomy_gap > 0.20:
            findings.append(
                f"Volume effect: adaptive runs at {a.mean_autonomy_utilization:.0%} autonomy vs "
                f"static {s.mean_autonomy_utilization:.0%} — {autonomy_gap:.0%} more decisions are "
                f"autonomous, increasing incident exposure proportionally. This is expected, "
                "not a quality regression."
            )

        # Efficiency
        findings.append(
            f"Efficiency (autonomy − incident rate): adaptive {a.mean_efficiency:.4f} vs "
            f"static {s.mean_efficiency:.4f} ({eff_gain:+.4f})."
        )

        # Anomaly vs normal period split
        if a.anomaly_steps > 0:
            a_anom_ir = a.mean_incident_rate_anomaly
            s_anom_ir = s.mean_incident_rate_anomaly
            anom_delta = ((s_anom_ir - a_anom_ir) / max(s_anom_ir, 1e-6)) * 100
            findings.append(
                f"During anomaly periods: adaptive IR {a_anom_ir:.4f} vs static IR {s_anom_ir:.4f} "
                f"({anom_delta:+.1f}% — {'adaptive contained it better' if anom_delta > 0 else 'static handled it better'})."
            )
            findings.append(
                f"During normal periods: adaptive efficiency {a.efficiency_during_normal:.4f} vs "
                f"static {s.efficiency_during_normal:.4f}."
            )

        # Boundary behavior
        b_moved = a.final_mean_boundary - a.initial_mean_boundary
        direction = "expanded" if b_moved > 0.02 else "contracted" if b_moved < -0.02 else "stayed stable"
        findings.append(
            f"Boundary {direction} from {a.initial_mean_boundary:.2f} → {a.final_mean_boundary:.2f} "
            f"(σ={a.boundary_stability_stddev:.3f})."
        )

        # Containment score (only meaningful for non-recovery scenarios)
        if scenario_name != "recovery":
            if a.containment_score > 0:
                findings.append(
                    f"Containment score: {a.containment_score:.2f} — adaptive suppressed "
                    f"{a.containment_score:.0%} of the incident surplus that static experienced during anomalies."
                )
            elif a.containment_score < -0.1:
                findings.append(
                    f"Containment score: {a.containment_score:.2f} — adaptive did WORSE than static "
                    "during anomaly periods, suggesting the boundary was over-expanded when stress hit."
                )

        # Re-expansion score for recovery (the right KPI for that scenario)
        if scenario_name == "recovery" and a.initial_mean_boundary < 0.40:
            re_expansion = max(
                0.0,
                (a.final_mean_boundary - a.initial_mean_boundary)
                / max(1.0 - a.initial_mean_boundary, 1e-6),
            )
            findings.append(
                f"Re-expansion score: {re_expansion:.2f} — system recovered "
                f"{re_expansion:.0%} of available autonomy headroom after conditions stabilised "
                f"(initial B={a.initial_mean_boundary:.2f} → final B={a.final_mean_boundary:.2f})."
            )

        # Decision-driven efficiency
        dde_delta = a.mean_decision_driven_efficiency - s.mean_decision_driven_efficiency
        findings.append(
            f"Decision-driven efficiency: adaptive {a.mean_decision_driven_efficiency:.1%} vs "
            f"static {s.mean_decision_driven_efficiency:.1%} ({dde_delta:+.1%}) — "
            f"fraction of ALL decisions handled autonomously AND correctly without human intervention."
        )
        findings.append(
            f"Autonomous success rate: {a.mean_autonomous_success_rate:.1%} of autonomous decisions "
            f"succeeded without incident/override/rollback "
            f"(normal: {a.dde_during_normal / max(a.mean_autonomy_utilization, 0.01):.1%}, "
            f"anomaly: {a.dde_during_anomaly / max(a.mean_autonomy_utilization, 0.01):.1%})."
        )
        if a.circuit_breaker_activations > 0:
            findings.append(
                f"Spike circuit breaker fired {a.circuit_breaker_activations} time(s) — "
                f"system jumped to conservative boundary on consistently-growing risk signals, "
                f"bypassing gradual contraction."
            )

        return findings

    # ── Takeaways (what it means) ─────────────────────────────────────────────

    def _takeaways_experiment(self, ir_delta, a, s) -> List[str]:
        takeaways = []
        au_diff = a.mean_autonomy_utilization - s.mean_autonomy_utilization

        if ir_delta > 5 and au_diff > 0.1:
            takeaways.append(
                "Adaptive is the better choice here: it achieves more automation AND safer "
                "outcomes. The β >> α asymmetry is working — contraction during risk, slow expansion during calm."
            )
        elif ir_delta > 0 and au_diff > 0.2:
            takeaways.append(
                "Adaptive trades a small safety improvement for significantly more automation throughput. "
                "Suitable for lower-risk decision classes where speed matters."
            )
        elif ir_delta < 0 and a.final_mean_boundary > 0.85:
            takeaways.append(
                "The boundary expanded to near-maximum without encountering enough anomaly pressure. "
                "In a production system, this means the system would be running highly automated just "
                "before a failure event — consider lowering α or setting a lower max_boundary."
            )
        elif ir_delta < -5:
            takeaways.append(
                "Static is safer here. Adaptive over-expanded because risk signals were too weak "
                "to trigger contraction. Either increase β, lower τ_risk, or ensure signal quality "
                "(are ROLLBACK/INCIDENT signals actually being emitted in proportion to failures?)."
            )

        if a.boundary_stability < 0.02:
            takeaways.append(
                "Boundary barely moved throughout the run. Possible causes: window_size too large "
                "(signals diluted), risk thresholds too high (S_t rarely reaches τ_risk or τ_safe), "
                "or the simulation anomaly rate is too low to generate meaningful feedback."
            )

        return takeaways

    def _takeaways_scenario(self, ir_delta, eff_gain, a, s, scenario_name) -> List[str]:
        takeaways = []

        # Scenario-specific interpretation
        context = SCENARIO_CONTEXT.get(scenario_name)
        if context:
            takeaways.append(f"What this scenario tests: {context}")

        # Outcome interpretation
        if scenario_name == "flash_sale":
            dde_gain = a.mean_decision_driven_efficiency - s.mean_decision_driven_efficiency
            if a.containment_score > 0.3:
                takeaways.append(
                    f"Adaptive successfully contained the spike: the boundary contracted fast enough "
                    "during the burst window to route high-risk decisions to human review. "
                    "The current β is sufficient for flash-sale-scale events."
                )
                if ir_delta < 0:
                    takeaways.append(
                        f"The overall IR gap ({abs(ir_delta):.1f}% more incidents) is a volume "
                        "effect — adaptive runs at ~90% autonomy vs static at ~45%. During the "
                        "actual spike, adaptive handled it better (positive containment). "
                        "DDE confirms: adaptive correctly processed more total decisions."
                    )
            elif a.containment_score > 0:
                takeaways.append(
                    "Partial containment: the boundary contracted but not fast enough to catch all "
                    "spike-period incidents. Consider increasing β or lowering τ_risk."
                )
            else:
                takeaways.append(
                    "Poor containment: the adaptive system matched or exceeded static incidents during "
                    "the spike. window_size may be too large relative to spike duration."
                )
            if dde_gain > 0.10:
                takeaways.append(
                    f"DDE gain of {dde_gain:+.1%} is the headline number: adaptive correctly handled "
                    f"{dde_gain:.0%} more decisions without human intervention. The system is "
                    "allocating human attention more efficiently than a fixed boundary."
                )

        elif scenario_name == "degradation":
            if ir_delta > 5:
                takeaways.append(
                    "Adaptive detected the slow-rolling degradation before it peaked — the OVERRIDE "
                    "and latency INCIDENT signals gave early warning. This validates §4.1 of the paper: "
                    "attribution-tolerant correlation is sufficient for gradual risk detection."
                )
            else:
                takeaways.append(
                    "Adaptive didn't clearly outperform static on gradual degradation. The slow error "
                    "climb may be below the signal threshold per window. Consider: lowering "
                    "moderate_error_threshold or high_latency_threshold_ms in the scenario config, "
                    "or increasing decisions_per_step to give more signal density per window."
                )

        elif scenario_name == "recovery":
            re_expansion = 0.0
            if a.initial_mean_boundary < 0.40:
                re_expansion = max(
                    0.0,
                    (a.final_mean_boundary - a.initial_mean_boundary)
                    / max(1.0 - a.initial_mean_boundary, 1e-6),
                )
            takeaways.append(
                "Note: incident rate comparison is not the right metric for recovery — the whole "
                "run is anomaly so static and adaptive are both operating under degraded conditions. "
                "The correct KPI is re-expansion score: did the system earn back autonomy correctly?"
            )
            if re_expansion >= 0.30:
                takeaways.append(
                    f"Re-expansion score {re_expansion:.2f} confirms the earned-autonomy mechanism "
                    "is working: boundary rebuilt correctly as safe windows accumulated. "
                    "α(1−B) is sufficient for post-incident recovery at current settings."
                )
            elif re_expansion >= 0.15:
                takeaways.append(
                    f"Partial re-expansion (score {re_expansion:.2f}): boundary is recovering but "
                    f"CB fired {a.circuit_breaker_activations}× — oscillation is slowing re-expansion. "
                    "Increasing sustained_safe_windows or safe_threshold would reduce CB re-triggers."
                )
            else:
                takeaways.append(
                    f"Low re-expansion (score {re_expansion:.2f}): system struggled to rebuild "
                    f"autonomy. CB fired {a.circuit_breaker_activations}× — sustained oscillation "
                    "is preventing stable re-expansion. Consider raising α slightly or tuning "
                    "recovery_lock_windows + sustained_safe_windows."
                )

        # General efficiency takeaway
        if eff_gain > 0.02:
            takeaways.append(
                f"Efficiency gain of {eff_gain:+.4f} means adaptive delivers more automation per "
                "unit of incident risk — the governance layer is allocating human attention "
                "more precisely than a fixed boundary would."
            )
        elif eff_gain < -0.02:
            takeaways.append(
                "Efficiency is lower in adaptive — the boundary adjustments are not yet "
                "saving enough incidents to justify the automation they're pulling back. "
                "The system is still in the conservative learning phase."
            )

        return takeaways

    # ── Warnings ──────────────────────────────────────────────────────────────

    def _warnings_experiment(self, ir_delta, a, s) -> List[str]:
        warnings = []
        if a.final_mean_boundary >= 0.85:
            warnings.append(
                f"Boundary reached {a.final_mean_boundary:.2f} — likely inside or above the headroom "
                "zone. The system has little contraction buffer remaining for unanticipated spikes. "
                "Consider increasing headroom_buffer (default 0.15) or lowering max_boundary."
            )
        if a.mean_autonomy_utilization > 0.85 and ir_delta < 0:
            warnings.append(
                "High automation + higher incident rate than static: classic over-expansion pattern. "
                "Increasing headroom_buffer would slow expansion near the ceiling and keep the "
                "boundary positioned to absorb a sudden risk event."
            )
        if a.boundary_stability < 0.005:
            warnings.append(
                "Boundary is essentially frozen (σ < 0.005). The feedback loop is not doing anything "
                "meaningful. Check that signals are being emitted and reaching the aggregator."
            )
        return warnings

    def _warnings_scenario(self, ir_delta, a, s, scenario_name: str = "") -> List[str]:
        warnings = []
        if a.final_mean_boundary >= 0.85:
            warnings.append(
                f"Boundary ended at {a.final_mean_boundary:.2f} — inside or above the headroom zone. "
                "If another event follows, the system enters it near fully-automated with limited "
                "room to contract. Use headroom_buffer to prevent the boundary from drifting this high "
                "during calm periods."
            )
        if a.containment_score < 0 and a.anomaly_steps > 0 and scenario_name != "recovery":
            warnings.append(
                "Negative containment: adaptive was WORSE than static during anomaly periods. "
                "Root cause: the boundary was already expanded into the headroom zone when the "
                "anomaly hit — contraction started from too high a value. A larger headroom_buffer "
                "would have kept the pre-event boundary lower."
            )
        if scenario_name == "recovery" and a.circuit_breaker_activations > 15:
            warnings.append(
                f"CB fired {a.circuit_breaker_activations}× in recovery — high oscillation. "
                "The system keeps expanding then getting hammered back to conservative boundary. "
                "Increase sustained_safe_windows (currently 3) to require a longer safety streak "
                "before expansion is permitted post-CB."
            )
        if a.min_efficiency < -0.3:
            warnings.append(
                f"Minimum per-step efficiency dropped to {a.min_efficiency:.3f} — at least one step "
                "combined high automation with high incident rate simultaneously. Review the "
                "system_behavior plot to identify which step this was."
            )
        return warnings

    # ── Printer ───────────────────────────────────────────────────────────────

    def print(self, insights: RunInsights, title: str = "Insights") -> None:
        """Render insights to stdout using Rich."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text
            console = Console()
        except ImportError:
            self._plain_print(insights)
            return

        verdict_colors = {
            "ADAPTIVE_WINS":        "bold green",
            "ADAPTIVE_EFFICIENT":   "bold green",
            "RECOVERY_HEALTHY":     "bold green",
            "STATIC_WINS":          "bold red",
            "RECOVERY_OSCILLATING": "bold yellow",
            "MIXED":                "bold yellow",
            "INCONCLUSIVE":         "bold white",
        }
        color = verdict_colors.get(insights.verdict, "bold white")

        lines = []
        lines.append(f"[{color}]Verdict: {insights.verdict}[/{color}]")
        lines.append(f"[italic]{insights.verdict_reason}[/italic]")

        if insights.findings:
            lines.append("\n[bold cyan]Key Findings[/bold cyan]")
            for f in insights.findings:
                lines.append(f"  • {f}")

        if insights.takeaways:
            lines.append("\n[bold cyan]Takeaways[/bold cyan]")
            for t in insights.takeaways:
                lines.append(f"  → {t}")

        if insights.warnings:
            lines.append("\n[bold red]Warnings[/bold red]")
            for w in insights.warnings:
                lines.append(f"  ⚠ {w}")

        console.print(Panel("\n".join(lines), title=f"[bold]{title}[/bold]", border_style="cyan"))

    def _plain_print(self, insights: RunInsights) -> None:
        print(f"\nVerdict: {insights.verdict}")
        print(f"{insights.verdict_reason}")
        if insights.findings:
            print("\nKey Findings:")
            for f in insights.findings:
                print(f"  • {f}")
        if insights.takeaways:
            print("\nTakeaways:")
            for t in insights.takeaways:
                print(f"  → {t}")
        if insights.warnings:
            print("\nWarnings:")
            for w in insights.warnings:
                print(f"  ⚠ {w}")
