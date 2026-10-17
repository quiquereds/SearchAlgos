"""Evaluación aislada de los métodos de referencia para el árbol de Steiner.

    python run_reference_methods.py                  # los 7 experimentos
    python run_reference_methods.py --only 1.1 2.1   # algunos
    python run_reference_methods.py --ilp-time-limit 1800

Corre Takahashi-Matsuyama y el ILP exacto sobre los mismos planos y conjuntos de demanda que
run_experiments.py, pero no ejecuta ni importa la búsqueda multi-demanda de Dijkstra: cada
evaluación se reporta por separado en results/reference_methods.*.
"""

import argparse
import json
import time
from pathlib import Path

from algorithms.steiner_solution import connects, edge_weights
from algorithms.takahashi_matsuyama import takahashi_matsuyama
from components.connectivity import DisconnectedGraphError
from components.route_format import format_tree
from problems.experiments import EXPERIMENTS, get_experiment


def evaluate(experiment, ilp_time_limit):
    from algorithms.steiner_ilp import steiner_ilp

    plan = experiment.plan
    root = plan.panel
    terminals = list(dict.fromkeys((root, *experiment.demand)))
    row = {
        "id": experiment.id,
        "plan": plan.name,
        "nodes": plan.node_count,
        "plan_edges": len(plan.edges),
        "demand_nodes": len(experiment.demand),
        "terminals": len(terminals),
        "human_reference_m": experiment.human_reference_m,
    }
    try:
        experiment.validate()
    except DisconnectedGraphError as error:
        return {**row, "status": "invalid", "reason": str(error)}

    started = time.perf_counter()
    heuristic = takahashi_matsuyama(plan.nodes, plan.edges, root, terminals)
    row["tm_time_s"] = time.perf_counter() - started
    row["tm_cost_m"] = heuristic.cost
    row["tm_valid"] = connects(heuristic, terminals)
    row["tm_edges"] = [list(edge) for edge in heuristic.edges]

    started = time.perf_counter()
    exact = steiner_ilp(plan.nodes, plan.edges, root, terminals, time_limit=ilp_time_limit)
    row["ilp_time_s"] = time.perf_counter() - started
    row["ilp_status"] = exact.status
    row["ilp_bound_m"] = exact.lower_bound
    if exact.solution is not None:
        row["ilp_cost_m"] = exact.solution.cost
        row["ilp_valid"] = connects(exact.solution, terminals)
        row["ilp_edges"] = [list(edge) for edge in exact.solution.edges]

    if exact.status == "optimal":
        optimum = exact.solution.cost
        row["tm_gap_pct"] = 100 * (heuristic.cost - optimum) / optimum
        row["human_gap_pct"] = 100 * (experiment.human_reference_m - optimum) / optimum
    row["status"] = "done"
    return row


def _format_duration(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)} m {rest:.0f} s"


def _cell(row, key, template):
    value = row.get(key)
    return "-" if value is None else format(value, template)


def render_table(rows):
    lines = [
        "| Exp | Plan | \\|D\\| | Humano (m) | TM (m) | TM tiempo | ILP (m) | Estado ILP | ILP tiempo "
        "| Brecha TM (%) | Brecha humano (%) |",
        "|" + " --- |" * 11,
    ]
    for row in rows:
        head = f"| {row['id']} | {row['plan']} | {row['demand_nodes']} |"
        if row["status"] != "done":
            lines.append(f"{head} {row['status']}: {row.get('reason', '')} |" + " - |" * 8)
            continue
        ilp_cost = _cell(row, "ilp_cost_m", ".2f")
        if row["ilp_status"] != "optimal" and row.get("ilp_bound_m") is not None:
            ilp_cost += f" (cota {row['ilp_bound_m']:.2f})"
        lines.append(
            f"{head} {row['human_reference_m']:g} | {row['tm_cost_m']:.2f} | {_format_duration(row['tm_time_s'])} "
            f"| {ilp_cost} | {row['ilp_status']} | {_format_duration(row['ilp_time_s'])} "
            f"| {_cell(row, 'tm_gap_pct', '.1f')} | {_cell(row, 'human_gap_pct', '.1f')} |"
        )
    return "\n".join(lines)


def render_route(row, experiment):
    weights = edge_weights(experiment.plan.edges)
    panel, demand = experiment.plan.panel, experiment.demand
    blocks = [
        f"Takahashi-Matsuyama (c = {row['tm_cost_m']:.2f} m, * = demanda):",
        format_tree(row["tm_edges"], panel, weights, demand),
    ]
    if "ilp_edges" in row:
        label = "ILP óptimo" if row["ilp_status"] == "optimal" else "ILP, mejor solución encontrada"
        blocks += [
            "",
            f"{label} (c = {row['ilp_cost_m']:.2f} m):",
            format_tree(row["ilp_edges"], panel, weights, demand),
        ]
    return "\n".join(blocks)


def render_routes_markdown(rows, experiments):
    sections = [
        f"### Exp {row['id']}\n\n```text\n{render_route(row, experiments[row['id']])}\n```"
        for row in rows
        if row["status"] == "done"
    ]
    return "## Rutas solución\n\n" + "\n\n".join(sections) if sections else ""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="ID", help="ids de experimento, por ejemplo 1.1 2.3")
    parser.add_argument(
        "--ilp-time-limit", type=float, default=600, help="segundos máximos del ILP por experimento (600)"
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
        row = evaluate(experiment, args.ilp_time_limit)
        rows.append(row)
        print(f"  -> {row['status']}", flush=True)
        if row["status"] == "done":
            print("\n" + render_route(row, experiment) + "\n", flush=True)

    table = render_table(rows)
    print("\n" + table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reference_methods.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    routes = render_routes_markdown(rows, {e.id: e for e in selected})
    (args.output_dir / "reference_methods.md").write_text(table + "\n\n" + routes + "\n")
    print(f"\nResultados en {args.output_dir}/")


if __name__ == "__main__":
    main()
