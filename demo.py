"""
Autonomy Learning Loop — Quick Demo

Runs three production scenarios (flash_sale, degradation, recovery) and
prints the key results: static vs adaptive comparison, per-scenario verdict,
and the cross-experiment optimization verdict.

Usage:
    python demo.py            # full demo (~30s)
    python demo.py --fast     # reduced steps (~10s)
"""
import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Fewer steps for a quicker run")
    args = parser.parse_args()

    steps = 150 if args.fast else 400

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.rule import Rule
        console = Console()
    except ImportError:
        print("Install rich: pip install rich")
        sys.exit(1)

    try:
        from src.simulation.scenarios import flash_sale_scenario, degradation_scenario, recovery_scenario
        from src.experiments.production_runner import ProductionExperimentRunner
        from src.experiments.optimizer import AutonomyOptimizer
        from src.experiments.insights import InsightsGenerator
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from the repo root: python demo.py")
        sys.exit(1)

    console.print()
    console.print(Rule("[bold cyan]Autonomy Learning Loop — Demo[/bold cyan]"))
    console.print(
        Panel(
            "[bold]What you're about to see:[/bold]\n\n"
            "  Three production scenarios — flash sale, gradual degradation, recovery.\n"
            "  Each runs [cyan]static[/cyan] (fixed 50% autonomy) vs [green]adaptive[/green] (feedback-driven boundary).\n"
            "  Key metric: [bold yellow]Decision-Driven Efficiency[/bold yellow] — "
            "fraction of ALL decisions handled\n"
            "  autonomously AND correctly without human intervention.",
            border_style="dim",
        )
    )
    console.print()

    # ── Governance parameters (tuned defaults) ────────────────────────────────
    params = dict(
        alpha=0.025,
        beta=0.38,
        safe_threshold=0.15,
        risk_threshold=0.60,
        headroom_buffer=0.35,
        window_size=min(35, steps // 5),
        seed=42,
    )

    runner = ProductionExperimentRunner(output_dir="outputs/demo")
    insights_gen = InsightsGenerator()
    results = {}

    scenarios = [
        ("flash_sale",   flash_sale_scenario(total_steps=steps, **{k: v for k, v in params.items() if k != "window_size"})),
        ("degradation",  degradation_scenario(total_steps=steps, **{k: v for k, v in params.items() if k != "window_size"})),
        ("recovery",     recovery_scenario(total_steps=steps, **{k: v for k, v in params.items() if k != "window_size"})),
    ]

    for name, scenario in scenarios:
        scenario.window_size = params["window_size"]
        console.print(f"[dim]Running scenario: [bold]{name.upper()}[/bold] ({steps} steps)...[/dim]")
        t0 = time.monotonic()
        result = runner.run_scenario(scenario, save_json=False)
        elapsed = time.monotonic() - t0
        results[name] = result

        # Print compact comparison
        a = result.adaptive_metrics
        s = result.static_metrics
        dde_gain = a.mean_decision_driven_efficiency - s.mean_decision_driven_efficiency

        console.print(
            f"  [green]✓[/green] {name:<14} "
            f"DDE: [yellow]{s.mean_decision_driven_efficiency:.0%}[/yellow] → [bold green]{a.mean_decision_driven_efficiency:.0%}[/bold green] "
            f"([bold]+{dde_gain:.0%}[/bold])  "
            f"Containment: [cyan]{a.containment_score:.2f}[/cyan]  "
            f"CB fired: [dim]{a.circuit_breaker_activations}×[/dim]  "
            f"[dim]{elapsed:.1f}s[/dim]"
        )

    # ── Per-scenario insights ─────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Scenario Verdicts[/bold]"))

    verdict_colors = {
        "ADAPTIVE_WINS":        "bold green",
        "ADAPTIVE_EFFICIENT":   "bold green",
        "RECOVERY_HEALTHY":     "bold green",
        "STATIC_WINS":          "bold red",
        "RECOVERY_OSCILLATING": "bold yellow",
        "MIXED":                "bold yellow",
        "INCONCLUSIVE":         "bold white",
    }

    for name, result in results.items():
        insights = insights_gen.from_scenario(result)
        color = verdict_colors.get(insights.verdict, "bold white")
        console.print(
            f"  [bold cyan]{name:<14}[/bold cyan]  "
            f"[{color}]{insights.verdict}[/{color}]\n"
            f"               [italic]{insights.verdict_reason}[/italic]"
        )

    # ── Cross-experiment optimizer ────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Cross-Experiment Optimization[/bold]"))

    optimizer = AutonomyOptimizer(current_params={**params, "max_boundary": 0.95})
    report = optimizer.analyze(scenario_results=results)

    verdict_panel_colors = {
        "ADAPTIVE_RECOMMENDED":           "bold green",
        "ADAPTIVE_EFFICIENT_TUNING_NEEDED": "bold yellow",
        "STRONG_ADAPTIVE_ADVANTAGE":      "bold green",
        "ADAPTIVE_ADVANTAGE_WITH_TUNING": "bold yellow",
        "REQUIRES_TUNING":                "bold yellow",
        "STATIC_PREFERRED_UNTIL_TUNED":   "bold red",
    }
    vc = verdict_panel_colors.get(report.overall_verdict, "bold white")
    console.print(
        Panel(
            f"[{vc}]{report.overall_verdict}[/{vc}]\n\n{report.overall_summary}",
            title="[bold]Overall Assessment[/bold]",
            border_style="cyan",
        )
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Key Numbers[/bold]"))

    from rich.table import Table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Scenario",          style="bold")
    table.add_column("Static DDE",        justify="right")
    table.add_column("Adaptive DDE",      justify="right")
    table.add_column("DDE Gain",          justify="right")
    table.add_column("Anomaly Contain.",  justify="right")
    table.add_column("Auto Success Rate", justify="right")

    for name, result in results.items():
        a = result.adaptive_metrics
        s = result.static_metrics
        dde_gain = a.mean_decision_driven_efficiency - s.mean_decision_driven_efficiency
        contain = f"{a.containment_score:.2f}" if name != "recovery" else "[dim]n/a*[/dim]"
        table.add_row(
            name,
            f"{s.mean_decision_driven_efficiency:.1%}",
            f"[bold green]{a.mean_decision_driven_efficiency:.1%}[/bold green]",
            f"[bold]+{dde_gain:.1%}[/bold]",
            contain,
            f"{a.mean_autonomous_success_rate:.1%}",
        )

    console.print(table)
    console.print("[dim]* Recovery scored on re-expansion, not containment (normal_IR ≈ 0)[/dim]")
    console.print()
    console.print(
        "[dim]To explore further:[/dim]\n"
        "  [cyan]python cli.py analyze[/cyan]              — full report + optimization recommendations\n"
        "  [cyan]python cli.py scenario flash_sale[/cyan]  — detailed plots + JSON\n"
        "  [cyan]python cli.py sweep --param alpha --values 0.01,0.025,0.05[/cyan]  — sensitivity analysis\n"
    )


if __name__ == "__main__":
    main()
