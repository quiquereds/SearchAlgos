from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from algorithms.steiner_solution import SteinerSolution, build_solution, edge_weights


@dataclass(frozen=True)
class IlpResult:
    status: str  # "optimal", "time_limit" o "infeasible"
    solution: SteinerSolution | None
    lower_bound: float | None
    mip_gap: float | None


def steiner_ilp(
    nodes: Sequence[str],
    edges: Iterable[tuple[str, str, float]],
    root: str,
    terminals: Iterable[str],
    time_limit: float | None = None,
) -> IlpResult:
    """Árbol de Steiner óptimo por programación lineal entera (HiGHS vía scipy).

    Formulación de flujo multi-commodity: una variable binaria y_e por conducto y, por cada
    terminal t distinta de la raíz, una unidad de flujo dirigida de la raíz a t que solo puede
    usar conductos con y_e = 1. Su cota lineal coincide con la del corte dirigido de Wong (1984).
    """
    edges = list(edges)
    weights = edge_weights(edges)
    node_index = {name: i for i, name in enumerate(nodes)}
    sinks = [t for t in dict.fromkeys(terminals) if t != root]
    if not sinks:
        return IlpResult("optimal", SteinerSolution(edges=(), cost=0.0), 0.0, 0.0)

    n_nodes, n_edges, n_sinks = len(node_index), len(edges), len(sinks)
    arcs = [(a, b) for a, b, _ in edges] + [(b, a) for a, b, _ in edges]
    n_arcs = len(arcs)
    n_vars = n_edges + n_sinks * n_arcs  # primero y_e, luego el flujo por terminal

    def flow_var(sink: int, arc: int) -> int:
        return n_edges + sink * n_arcs + arc

    # Conservación de flujo: salida - entrada = +1 en la raíz, -1 en la terminal, 0 en el resto.
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_sinks * n_nodes)
    for s, sink in enumerate(sinks):
        base = s * n_nodes
        for k, (u, v) in enumerate(arcs):
            rows += [base + node_index[u], base + node_index[v]]
            cols += [flow_var(s, k)] * 2
            vals += [1.0, -1.0]
        rhs[base + node_index[root]] = 1.0
        rhs[base + node_index[sink]] = -1.0
    conservation = coo_matrix((vals, (rows, cols)), shape=(n_sinks * n_nodes, n_vars)).tocsr()

    # El flujo de cada terminal por un arco no supera la variable y del conducto.
    rows, cols, vals = [], [], []
    for s in range(n_sinks):
        for k in range(n_arcs):
            row = s * n_arcs + k
            rows += [row, row]
            cols += [flow_var(s, k), k % n_edges]
            vals += [1.0, -1.0]
    capacity = coo_matrix((vals, (rows, cols)), shape=(n_sinks * n_arcs, n_vars)).tocsr()

    cost = np.zeros(n_vars)
    cost[:n_edges] = [weight for _, _, weight in edges]
    integrality = np.zeros(n_vars)
    integrality[:n_edges] = 1
    options: dict[str, object] = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = time_limit

    result = milp(
        cost,
        constraints=[
            LinearConstraint(conservation, rhs, rhs),
            LinearConstraint(capacity, -np.inf, 0.0),
        ],
        integrality=integrality,
        bounds=Bounds(0.0, 1.0),
        options=options,
    )

    bound = getattr(result, "mip_dual_bound", None)
    gap = getattr(result, "mip_gap", None)
    if result.x is None:
        status = "infeasible" if result.status == 2 else "time_limit"
        return IlpResult(status, None, bound, gap)

    chosen = [(a, b) for k, (a, b, _) in enumerate(edges) if result.x[k] > 0.5]
    status = "optimal" if result.status == 0 else "time_limit"
    return IlpResult(status, build_solution(chosen, weights), bound, gap)
