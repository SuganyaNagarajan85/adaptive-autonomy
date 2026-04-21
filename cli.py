"""
CLI — command-line interface for running simulations and experiments.

Usage examples:
  python cli.py run --steps 500 --adaptive
  python cli.py experiment --steps 300 --alpha 0.05 --beta 0.30 --plot
  python cli.py status
  python cli.py sweep --param alpha --values 0.02,0.05,0.10 --steps 200
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()

logging.basicConfig(
    level=logging.WARNING,
    handlers=[RichHandler(console=console, show_path=False)],
    format="%(message)s",
)


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── CLI Group ───────────────────────────────────────────────────────────────
@click.group()
@click.option(
    "--config", default="config/ecommerce.yaml",
    show_default=True, help="Path to YAML config file"
)
@click.pass_context
def cli(ctx, config):
    """Feedback-Driven Autonomy Learning Loop — CLI"""
    ctx.ensure_object(dict)
    try:
        ctx.obj["config"] = _load_config(config)
    except FileNotFoundError:
        ctx.obj["config"] = {}
    ctx.obj["config_path"] = config


# ─── run ─────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--steps", default=500, show_default=True, help="Simulation steps")
@click.option("--decisions", default=20, show_default=True, help="Decisions per step")
@click.option("--adaptive/--static", default=True, help="Enable/disable feedback loop")
@click.option("--alpha", default=0.05, show_default=True, help="Expansion rate α")
@click.option("--beta", default=0.30, show_default=True, help="Contraction rate β")
@click.option("--initial-boundary", default=0.5, show_default=True, help="Initial B₀")
@click.option("--seed", default=42, show_default=True, help="Random seed")
@click.option("--plot/--no-plot", default=True, help="Save visualization plots")
@click.option("--output-dir", default="outputs", show_default=True, help="Output directory")
@click.pass_context
def run(ctx, steps, decisions, adaptive, alpha, beta, initial_boundary, seed, plot, output_dir):
    """Run a single simulation (adaptive or static)."""
    from src.simulation.simulator import Simulator

    mode = "ADAPTIVE" if adaptive else "STATIC"
    console.print(f"\n[bold cyan]Running {mode} simulation[/bold cyan]")
    console.print(f"  Steps: {steps} | Decisions/step: {decisions} | B₀: {initial_boundary}")
    console.print(f"  α={alpha}  β={beta}  seed={seed}\n")

    sim = Simulator(
        total_steps=steps,
        decisions_per_step=decisions,
        adaptive=adaptive,
        alpha=alpha,
        beta=beta,
        initial_boundary=initial_boundary,
        seed=seed,
        min_update_interval_seconds=0.0,
    )

    with console.status("[bold green]Simulating..."):
        step_metrics = sim.run()

    summary = sim.metrics_collector.summarize(label=mode.lower())

    # Print summary table
    table = Table(title=f"{mode} Simulation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for k, v in summary.to_dict().items():
        if k == "label":
            continue
        table.add_row(k, str(round(v, 4) if isinstance(v, float) else v))

    console.print(table)

    if plot:
        _save_single_run_plots(step_metrics, output_dir, mode.lower())
        console.print(f"\n[green]Plots saved to: {output_dir}/[/green]")


# ─── experiment ──────────────────────────────────────────────────────────────
@cli.command()
@click.option("--steps", default=500, show_default=True)
@click.option("--decisions", default=20, show_default=True)
@click.option("--alpha", default=0.05, show_default=True)
@click.option("--beta", default=0.30, show_default=True)
@click.option("--safe-threshold", default=0.25, show_default=True)
@click.option("--risk-threshold", default=0.60, show_default=True)
@click.option("--initial-boundary", default=0.5, show_default=True)
@click.option("--anomaly-prob", default=0.05, show_default=True)
@click.option("--failure-rate", default=0.08, show_default=True)
@click.option("--headroom", default=0.15, show_default=True, help="Headroom buffer below max_boundary")
@click.option("--seed", default=42, show_default=True)
@click.option("--plot/--no-plot", default=True)
@click.option("--output-dir", default="outputs", show_default=True)
@click.option("--save-json", default=None, help="Save results to JSON file")
@click.pass_context
def experiment(
    ctx, steps, decisions, alpha, beta, safe_threshold, risk_threshold,
    initial_boundary, anomaly_prob, failure_rate, headroom, seed, plot, output_dir, save_json
):
    """Run static vs adaptive comparison experiment."""
    from src.experiments.runner import ExperimentRunner
    from src.experiments.visualizer import Visualizer

    if alpha >= beta:
        console.print("[red]Error: β must be greater than α[/red]")
        sys.exit(1)

    console.print("\n[bold cyan]Running Static vs Adaptive Comparison Experiment[/bold cyan]")
    console.print(f"  Steps: {steps} | Decisions/step: {decisions}")
    console.print(f"  α={alpha}  β={beta}  τ_safe={safe_threshold}  τ_risk={risk_threshold}")
    console.print(f"  Anomaly prob: {anomaly_prob}  Failure rate: {failure_rate}  Headroom: {headroom}\n")

    runner = ExperimentRunner()
    with console.status("[bold green]Running simulations..."):
        result = runner.run_comparison(
            total_steps=steps,
            decisions_per_step=decisions,
            alpha=alpha,
            beta=beta,
            safe_threshold=safe_threshold,
            risk_threshold=risk_threshold,
            initial_boundary=initial_boundary,
            anomaly_probability=anomaly_prob,
            base_failure_rate=failure_rate,
            headroom_buffer=headroom,
            seed=seed,
        )

    result.print_comparison()

    from src.experiments.insights import InsightsGenerator
    insights = InsightsGenerator().from_experiment(result)
    InsightsGenerator().print(insights, title="Experiment Insights")

    if plot:
        viz = Visualizer(output_dir=output_dir)
        paths = viz.plot_all(result)
        console.print(f"\n[green]Plots saved:[/green]")
        for p in paths:
            console.print(f"  → {p}")

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        console.print(f"\n[green]Results JSON saved to: {save_json}[/green]")


# ─── sweep ───────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--param", required=True, help="Parameter to sweep (alpha, beta, safe_threshold, ...)")
@click.option("--values", required=True, help="Comma-separated values to try")
@click.option("--steps", default=300, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--output-dir", default="outputs/sweep", show_default=True)
def sweep(param, values, steps, seed, output_dir):
    """Sensitivity sweep: vary one parameter across values."""
    from src.experiments.runner import ExperimentRunner

    param_values = [float(v) for v in values.split(",")]
    console.print(f"\n[bold cyan]Sensitivity Sweep: {param} = {param_values}[/bold cyan]\n")

    runner = ExperimentRunner()
    results = runner.run_sensitivity_sweep(
        param_name=param,
        param_values=param_values,
        total_steps=steps,
        seed=seed,
    )

    table = Table(title=f"Sensitivity Sweep: {param}", show_header=True)
    table.add_column(param, style="cyan")
    table.add_column("Incident Reduction %", justify="right")
    table.add_column("Override Reduction %", justify="right")
    table.add_column("Adaptive Incident Rate", justify="right")
    table.add_column("Boundary Stability σ", justify="right")

    for val, r in zip(param_values, results):
        table.add_row(
            str(val),
            f"{r.incident_reduction_pct:+.1f}%",
            f"{r.override_reduction_pct:+.1f}%",
            f"{r.adaptive_summary.mean_incident_rate:.4f}",
            f"{r.adaptive_summary.boundary_stability:.4f}",
        )

    console.print(table)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sweep_data = [
        {"param_value": v, **r.to_dict()}
        for v, r in zip(param_values, results)
    ]
    out_path = f"{output_dir}/{param}_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep_data, f, indent=2, default=str)
    console.print(f"\n[green]Sweep results saved to: {out_path}[/green]")


# ─── scenario ────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("scenario_name", type=click.Choice(["flash_sale", "degradation", "recovery", "all"]))
@click.option("--steps", default=None, type=int, show_default=True,
              help="Override total_steps (uses scenario default if omitted)")
@click.option("--plot/--no-plot", default=True, help="Generate and save plots")
@click.option("--output-dir", default="outputs", show_default=True, help="Output base directory")
@click.option("--save-json/--no-save-json", default=True, help="Save results as JSON")
@click.option("--seed", default=None, type=int, help="Override scenario seed")
def scenario(scenario_name, steps, plot, output_dir, save_json, seed):
    """
    Run a production scenario experiment (flash_sale | degradation | recovery | all).

    Examples:

        python cli.py scenario flash_sale --steps 400 --plot

        python cli.py scenario all --output-dir outputs/production
    """
    from src.experiments.production_runner import ProductionExperimentRunner
    from src.experiments.production_visualizer import ProductionVisualizer
    from src.simulation.scenarios import SCENARIO_REGISTRY, get_scenario

    runner = ProductionExperimentRunner(output_dir=output_dir)
    viz    = ProductionVisualizer(output_dir=output_dir)

    overrides = {}
    if steps is not None:
        overrides["total_steps"] = steps
    if seed is not None:
        overrides["seed"] = seed

    scenario_names = list(SCENARIO_REGISTRY.keys()) if scenario_name == "all" else [scenario_name]

    for sname in scenario_names:
        console.print(f"\n[bold cyan]Running scenario: {sname.upper()}[/bold cyan]")

        sc = get_scenario(sname, **overrides)
        console.print(f"  {sc.description}")
        console.print(f"  Steps: {sc.total_steps} | Decisions/step: {sc.decisions_per_step}")
        console.print(f"  α={sc.alpha}  β={sc.beta}  τ_safe={sc.safe_threshold}  τ_risk={sc.risk_threshold}\n")

        with console.status(f"[bold green]Simulating {sname}..."):
            result = runner.run_scenario(sc, save_json=save_json)

        result.print_comparison()

        from src.experiments.insights import InsightsGenerator
        insights = InsightsGenerator().from_scenario(result)
        InsightsGenerator().print(insights, title=f"{sname.upper()} Insights")

        if plot:
            with console.status("[bold green]Generating plots..."):
                paths = viz.plot_all(result)
            console.print(f"\n[green]Plots saved:[/green]")
            for p in paths:
                console.print(f"  → {p}")


# ─── analyze ─────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--steps", default=400, show_default=True, help="Steps per scenario")
@click.option("--alpha", default=0.025, show_default=True)
@click.option("--beta", default=0.38, show_default=True)
@click.option("--safe-threshold", default=0.15, show_default=True)
@click.option("--risk-threshold", default=0.60, show_default=True)
@click.option("--headroom", default=0.35, show_default=True)
@click.option("--window-size", default=35, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--output-dir", default="outputs/analysis", show_default=True)
@click.option("--save-json/--no-save-json", default=True)
def analyze(steps, alpha, beta, safe_threshold, risk_threshold, headroom, window_size, seed, output_dir, save_json):
    """
    Run all scenarios + baseline experiment, then produce a unified
    cross-experiment optimization report with parameter recommendations.

    Example:

        python cli.py analyze --steps 400 --headroom 0.35
    """
    import json
    from pathlib import Path
    from src.experiments.optimizer import AutonomyOptimizer
    from src.experiments.production_runner import ProductionExperimentRunner
    from src.experiments.runner import ExperimentRunner
    from src.simulation.scenarios import SCENARIO_REGISTRY, get_scenario

    current_params = {
        "alpha": alpha,
        "beta": beta,
        "safe_threshold": safe_threshold,
        "risk_threshold": risk_threshold,
        "headroom_buffer": headroom,
        "window_size": window_size,
        "max_boundary": 0.95,
    }

    console.print(f"\n[bold cyan]Full Autonomy Analysis[/bold cyan]")
    console.print(f"  α={alpha}  β={beta}  τ_safe={safe_threshold}  τ_risk={risk_threshold}")
    console.print(f"  headroom={headroom}  window_size={window_size}  seed={seed}\n")

    # ── 1. Baseline experiment (synthetic anomalies) ───────────────────────
    console.print("[bold]Step 1/5:[/bold] Running baseline experiment (synthetic anomalies)...")
    base_runner = ExperimentRunner()
    with console.status("Simulating baseline..."):
        experiment_result = base_runner.run_comparison(
            total_steps=steps,
            alpha=alpha, beta=beta,
            safe_threshold=safe_threshold,
            risk_threshold=risk_threshold,
            headroom_buffer=headroom,
            window_size=window_size,
            seed=seed,
        )
    experiment_result.print_comparison()

    # ── 2–4. Production scenarios ──────────────────────────────────────────
    prod_runner = ProductionExperimentRunner(output_dir=output_dir)
    scenario_results = {}
    scenario_names = list(SCENARIO_REGISTRY.keys())

    for i, sname in enumerate(scenario_names, start=2):
        console.print(f"\n[bold]Step {i}/5:[/bold] Running scenario: [cyan]{sname.upper()}[/cyan]")
        sc = get_scenario(
            sname,
            total_steps=steps,
            alpha=alpha, beta=beta,
            safe_threshold=safe_threshold,
            risk_threshold=risk_threshold,
            headroom_buffer=headroom,
            window_size=window_size,
            seed=seed,
        )
        console.print(f"  {sc.description}")
        with console.status(f"Simulating {sname}..."):
            result = prod_runner.run_scenario(sc, save_json=save_json)
        result.print_comparison()
        scenario_results[sname] = result

    # ── 5. Per-scenario insights ────────────────────────────────────────────
    console.print(f"\n[bold]Step 5/5:[/bold] Computing per-scenario insights...")
    from src.experiments.insights import InsightsGenerator
    gen = InsightsGenerator()
    for sname, result in scenario_results.items():
        insights = gen.from_scenario(result)
        gen.print(insights, title=f"{sname.upper()} Insights")

    # ── Optimization report ─────────────────────────────────────────────────
    console.print("\n[bold cyan]━━━ Optimization Report ━━━[/bold cyan]\n")
    optimizer = AutonomyOptimizer(current_params=current_params)
    report = optimizer.analyze(
        scenario_results=scenario_results,
        experiment_result=experiment_result,
    )
    optimizer.print_report(report)

    # ── Save report JSON ───────────────────────────────────────────────────
    if save_json:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = f"{output_dir}/optimization_report.json"
        report_dict = {
            "current_params": report.current_params,
            "recommended_params": report.recommended_params,
            "overall_verdict": report.overall_verdict,
            "overall_summary": report.overall_summary,
            "cross_scenario_findings": report.cross_scenario_findings,
            "anomaly_profiles": {
                n: {
                    "intensity_ratio": p.intensity_ratio,
                    "pre_event_boundary": p.pre_event_boundary,
                    "boundary_depth": p.boundary_depth,
                    "containment_score": p.containment_score,
                    "peak_incident_rate": p.peak_incident_rate,
                    "anomaly_steps": p.anomaly_steps,
                    "re_expansion_score": p.re_expansion_score,
                    "pathological_intensity": p.pathological_intensity,
                }
                for n, p in report.anomaly_profiles.items()
            },
            "recommendations": [
                {
                    "parameter": r.parameter,
                    "current": r.current_value,
                    "recommended": r.recommended_value,
                    "direction": r.direction,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "evidence": r.evidence,
                }
                for r in report.recommendations
            ],
        }
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        console.print(f"\n[green]Optimization report saved to: {out_path}[/green]")

    # ── Quick re-run hint ──────────────────────────────────────────────────
    # Map internal param names to CLI flag names
    _param_to_flag = {
        "alpha": "alpha",
        "beta": "beta",
        "risk_threshold": "risk-threshold",
        "safe_threshold": "safe-threshold",
        "headroom_buffer": "headroom",
        "window_size": "window-size",
    }
    changed = {k: v for k, v in report.recommended_params.items() if v != report.current_params.get(k)}
    if changed:
        flags = " ".join(
            f"--{_param_to_flag[k]} {v}"
            for k, v in changed.items()
            if k in _param_to_flag
        )
        console.print(f"\n[bold]Re-run with optimized parameters:[/bold]")
        console.print(f"  [cyan]python cli.py analyze {flags}[/cyan]")


# ─── status ──────────────────────────────────────────────────────────────────
@cli.command()
def status():
    """Show current boundary states from the controller."""
    from src.autonomy_controller.controller import AutonomyController

    controller = AutonomyController(initial_boundary=0.5)

    table = Table(title="Current Autonomy Boundary States", show_header=True)
    table.add_column("Decision Type", style="cyan")
    table.add_column("Boundary B", justify="right")
    table.add_column("Frozen", justify="center")

    for dt in controller._states:
        state = controller.get_state(dt)
        table.add_row(
            dt.value,
            f"{state.boundary:.3f}",
            "🔒" if state.frozen else "✅",
        )
    console.print(table)


def _save_single_run_plots(step_metrics, output_dir: str, label: str) -> None:
    """Save basic boundary + incident plots for a single simulation run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.decision_engine.models import DecisionType

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    x = [s.step for s in step_metrics]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for dt in DecisionType:
        series = [s.boundary_snapshots.get(dt.value, 0.0) for s in step_metrics]
        axes[0].plot(x, series, label=dt.value.replace("_", " ").title(), linewidth=1.5)

    axes[0].set_ylabel("Autonomy Boundary B")
    axes[0].set_title(f"{label.upper()} — Boundary Evolution")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    ir = [s.incident_rate for s in step_metrics]
    axes[1].plot(x, ir, color="#E84855", linewidth=1.5, label="Incident Rate")
    axes[1].set_ylabel("Incident Rate")
    axes[1].set_xlabel("Step")
    axes[1].set_title("Incident Rate Over Time")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = f"{output_dir}/{label}_run.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cli()
