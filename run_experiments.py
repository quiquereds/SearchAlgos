"""Ejecuta los experimentos de la adaptación multi-demanda de Dijkstra.

    python run_experiments.py                    # todos los experimentos
    python run_experiments.py --only 1.1 2.1     # solo algunos
    python run_experiments.py --exact            # añade el óptimo c* (hasta 16 terminales)

Cada experimento corre en un proceso aparte para aislar su memoria y poder cortarlo al
superar el límite de tiempo o de memoria; en ese caso se reporta como "No completado".
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
import tracemalloc
from pathlib import Path

import psutil

from algorithms.dijkstra_multi_goal import dijkstra_multi_goal
from algorithms.steiner_solution import edge_weights
from components.connectivity import DisconnectedGraphError
from components.route_format import format_tree, format_walk
from problems.experiments import EXPERIMENTS, get_experiment

_POLL_SECONDS = 0.5
_BYTES_PER_MB = 1024 * 1024


def _worker(experiment_id, trace_memory, expanded, current_g, connection):
    try:
        problem = get_experiment(experiment_id).build().create_multi_objective_problem()

        def on_progress(count, g):
            expanded.value = count
            current_g.value = g

        if trace_memory:
            tracemalloc.start()
        started = time.perf_counter()
        result = dijkstra_multi_goal(problem, progress_callback=on_progress)
        elapsed = time.perf_counter() - started
        traced_peak = None
        if trace_memory:
            traced_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

        payload = {"status": "no_solution" if result is None else "solved", "time_s": elapsed}
        payload["end_rss_bytes"] = psutil.Process().memory_info().rss
        payload["peak_traced_mb"] = None if traced_peak is None else traced_peak / _BYTES_PER_MB
        if result is not None:
            payload.update(
                cost_m=result.cost,
                guiding_cost_m=result.guiding_cost,
                states_expanded=result.states_expanded,
                is_tree=result.is_tree,
                walk=list(result.walk),
                tree_edges=[list(edge) for edge in result.edges],
            )
        connection.send(payload)
    except Exception as error:
        connection.send({"status": "error", "reason": f"{type(error).__name__}: {error}"})


def run_experiment(experiment, trace_memory, time_limit, memory_limit_bytes, show_progress):
    context = mp.get_context("spawn")
    expanded = context.Value("Q", 0, lock=False)
    current_g = context.Value("d", 0.0, lock=False)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(experiment.id, trace_memory, expanded, current_g, sender))

    started = time.perf_counter()
    process.start()
    sender.close()
    watcher = psutil.Process(process.pid)
    peak_rss = 0
    stop_reason = None

    while process.is_alive():
        process.join(_POLL_SECONDS)
        elapsed = time.perf_counter() - started
        try:
            peak_rss = max(peak_rss, watcher.memory_info().rss)
        except psutil.NoSuchProcess:
            break
        if time_limit and elapsed > time_limit:
            stop_reason = f"límite de tiempo ({_format_duration(time_limit)})"
        elif memory_limit_bytes and peak_rss > memory_limit_bytes:
            stop_reason = f"límite de memoria ({memory_limit_bytes / 1024**3:g} GB)"
        if stop_reason:
            process.terminate()
            process.join()
            break
        if show_progress and process.is_alive():
            print(
                f"\r  [{experiment.id}] {_format_duration(elapsed)} | "
                f"estados ~{expanded.value:,} | RSS {peak_rss / _BYTES_PER_MB:,.0f} MB   ",
                end="",
                flush=True,
            )
    if show_progress:
        print("\r" + " " * 78 + "\r", end="")

    elapsed = time.perf_counter() - started
    if stop_reason:
        return {
            "status": "not_completed",
            "reason": stop_reason,
            "time_s": elapsed,
            "states_expanded": expanded.value,
            "peak_rss_mb": peak_rss / _BYTES_PER_MB,
        }
    if not receiver.poll():
        return {"status": "error", "reason": f"el proceso terminó con código {process.exitcode}"}
    payload = receiver.recv()
    peak_rss = max(peak_rss, payload.pop("end_rss_bytes", 0))
    payload["peak_rss_mb"] = peak_rss / _BYTES_PER_MB
    return payload


def evaluate(experiment, args):
    row = {
        "id": experiment.id,
        "plan": experiment.plan.name,
        "nodes": experiment.plan.node_count,
        "plan_edges": len(experiment.plan.edges),
        "demand_nodes": len(experiment.demand),
        "human_reference_m": experiment.human_reference_m,
    }
    try:
        experiment.validate()
    except DisconnectedGraphError as error:
        return {**row, "status": "invalid", "reason": str(error)}

    memory_limit = int(args.memory_limit_gb * 1024**3)
    row.update(run_experiment(experiment, False, args.time_limit, memory_limit, sys.stdout.isatty()))
    if row["status"] != "solved":
        return row

    row["delta_pct"] = 100 * (row["cost_m"] - experiment.human_reference_m) / experiment.human_reference_m
    if args.trace_memory:
        traced = run_experiment(experiment, True, args.time_limit, memory_limit, False)
        row["peak_traced_mb"] = traced.get("peak_traced_mb")
    if args.exact:
        row.update(_exact_reference(experiment, row["cost_m"]))
    return row


def _exact_reference(experiment, cost):
    from algorithms.steiner_exact import MAX_TERMINALS, steiner_tree_cost

    terminals = list(dict.fromkeys((experiment.plan.panel, *experiment.demand)))
    if len(terminals) > MAX_TERMINALS:
        return {"exact_cost_m": None, "exact_note": f"{len(terminals)} terminales > {MAX_TERMINALS}"}
    started = time.perf_counter()
    optimum = steiner_tree_cost(experiment.plan.nodes, experiment.plan.edges, terminals)
    return {
        "exact_cost_m": optimum,
        "gap_pct": 100 * (cost - optimum) / optimum,
        "exact_time_s": time.perf_counter() - started,
    }


def _format_duration(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} m {rest:.0f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} m"


def _format_memory(megabytes):
    return f"{megabytes / 1024:.2f} GB" if megabytes >= 1024 else f"{megabytes:.1f} MB"


def render_table(rows):
    header = (
        "| Exp | Plan | \\|D\\| | Costo c (m) | g* (m) | Humano (m) | Δ (%) | c* (m) | Brecha (%) "
        "| Tiempo | Estados | Memoria pico | Árbol |"
    )
    lines = [header, "|" + " --- |" * 13]
    for row in rows:
        head = f"| {row['id']} | {row['plan']} | {row['demand_nodes']} |"
        if row["status"] == "solved":
            exact = row.get("exact_cost_m")
            lines.append(
                f"{head} {row['cost_m']:.2f} | {row['guiding_cost_m']:.2f} | {row['human_reference_m']:g} "
                f"| {row['delta_pct']:+.1f} | {'-' if exact is None else f'{exact:.2f}'} "
                f"| {'-' if exact is None else format(row['gap_pct'], '.1f')} "
                f"| {_format_duration(row['time_s'])} | {row['states_expanded']:,} "
                f"| {_format_memory(row['peak_rss_mb'])} | {'sí' if row['is_tree'] else 'no'} |"
            )
        elif row["status"] == "not_completed":
            lines.append(
                f"{head} No completado | - | {row['human_reference_m']:g} | - | - | - "
                f"| {_format_duration(row['time_s'])} (cortado: {row['reason']}) "
                f"| ≥ {row['states_expanded']:,} | {_format_memory(row['peak_rss_mb'])} | - |"
            )
        else:
            lines.append(f"{head} {row['status']}: {row.get('reason', '')} |" + " - |" * 9)
    return "\n".join(lines)


def render_route(row, experiment):
    weights = edge_weights(experiment.plan.edges)
    return "\n".join(
        [
            f"Recorrido de la búsqueda ({len(row['walk']) - 1} tramos, g* = {row['guiding_cost_m']:.2f} m):",
            format_walk(row["walk"]),
            "",
            f"Árbol de cable (c = {row['cost_m']:.2f} m, * = demanda):",
            format_tree(row["tree_edges"], experiment.plan.panel, weights, experiment.demand),
        ]
    )


def render_routes_markdown(rows, experiments):
    sections = [
        f"### Exp {row['id']}\n\n```text\n{render_route(row, experiments[row['id']])}\n```"
        for row in rows
        if row["status"] == "solved"
    ]
    return "## Rutas solución\n\n" + "\n\n".join(sections) if sections else ""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="ID", help="ids de experimento, por ejemplo 1.1 2.3")
    parser.add_argument("--exact", action="store_true", help="calcula el óptimo c* de Steiner cuando |D ∪ {s}| <= 16")
    parser.add_argument(
        "--trace-memory", action="store_true", help="repite la corrida con tracemalloc para el pico de Python"
    )
    parser.add_argument(
        "--time-limit", type=float, default=3600, help="segundos por experimento (3600); 0 = sin límite"
    )
    parser.add_argument(
        "--memory-limit-gb", type=float, default=16, help="RSS máximo por experimento en GB (16); 0 = sin límite"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    selected = [get_experiment(i) for i in args.only] if args.only else list(EXPERIMENTS)

    rows = []
    for experiment in selected:
        print(
            f"Experimento {experiment.id}: plano {experiment.plan.name}, {len(experiment.demand)} demandas", flush=True
        )
        row = evaluate(experiment, args)
        rows.append(row)
        print(f"  -> {row['status']}" + (f" ({row['reason']})" if row.get("reason") else ""), flush=True)
        if row["status"] == "solved":
            print("\n" + render_route(row, experiment) + "\n", flush=True)

    table = render_table(rows)
    print("\n" + table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiments.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    routes = render_routes_markdown(rows, {e.id: e for e in selected})
    (args.output_dir / "experiments.md").write_text(table + "\n\n" + routes + "\n")
    print(f"\nResultados en {args.output_dir}/")


if __name__ == "__main__":
    main()
