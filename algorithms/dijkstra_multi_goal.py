from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush

from components.connectivity import unreachable_targets
from components.problem import Problem

# Cada cuántas expansiones se invoca progress_callback (potencia de dos menos uno).
_PROGRESS_MASK = 0x3FFF


@dataclass(frozen=True)
class MultiDemandResult:
    """Solución de la búsqueda multi-demanda.

    cost es el costo reportado c(W*): suma de las aristas distintas de la caminata.
    guiding_cost es g*, el costo de la caminata contando cada cruce de arista.
    Se cumple c* <= cost <= guiding_cost <= 2 c* (Proposición 2 del artículo).
    """

    walk: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    cost: float
    guiding_cost: float
    states_expanded: int

    @property
    def is_tree(self) -> bool:
        """True si las aristas distintas no forman ciclos."""
        nodes = {name for edge in self.edges for name in edge}
        return len(self.edges) == max(len(nodes) - 1, 0)


def dijkstra_multi_goal(
    problem: Problem,
    progress_callback: Callable[[int, float], None] | None = None,
) -> MultiDemandResult | None:
    """Dijkstra multi-demanda (Algoritmo 1 del artículo).

    El estado de búsqueda es (v, S): vértice actual y conjunto de demandas ya conectadas,
    codificado como máscara de bits. La cola se ordena por g, la suma de pesos de la caminata,
    y la búsqueda termina al extraer el primer estado con S = D, que es óptimo para g.
    El costo reportado cuenta cada arista distinta una sola vez.

    Devuelve None si alguna demanda es inalcanzable desde el estado inicial.
    progress_callback recibe (estados expandidos, g actual).
    """
    if unreachable_targets(problem):
        return None

    names = list(problem.actions)
    index = {name: i for i, name in enumerate(names)}
    demand_bit: dict[int, int] = {}
    for target in problem.targets:
        demand_bit.setdefault(index[target.name], 1 << len(demand_bit))
    full_mask = (1 << len(demand_bit)) - 1

    adjacency: list[tuple[tuple[int, float, int], ...]] = []
    for name in names:
        neighbours = problem.actions[name]
        adjacency.append(
            tuple(
                (index[state.name], problem.costs[name][action], demand_bit.get(index[state.name], 0))
                for action, state in neighbours.items()
            )
        )

    # Estado empaquetado en un entero: la máscara en los bits altos y el vértice en los bajos.
    shift = len(names).bit_length()
    vertex_mask = (1 << shift) - 1

    start = index[problem.initial.name]
    start_key = (demand_bit.get(start, 0) << shift) | start
    best: dict[int, float] = {start_key: 0.0}
    parent: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, start_key)]
    expanded = 0

    while heap:
        g, key = heappop(heap)
        if g > best[key]:
            continue
        expanded += 1
        if progress_callback is not None and not expanded & _PROGRESS_MASK:
            progress_callback(expanded, g)

        covered = key >> shift
        if covered == full_mask:
            return _reconstruct(key, g, expanded, parent, vertex_mask, names, adjacency)

        vertex = key & vertex_mask
        for neighbour, weight, bit in adjacency[vertex]:
            new_g = g + weight
            new_key = ((covered | bit) << shift) | neighbour
            known = best.get(new_key)
            if known is None or new_g < known:
                best[new_key] = new_g
                parent[new_key] = key
                heappush(heap, (new_g, new_key))
    return None


def _reconstruct(goal_key, guiding_cost, expanded, parent, vertex_mask, names, adjacency):
    walk_indices = [goal_key & vertex_mask]
    key = goal_key
    while key in parent:
        key = parent[key]
        walk_indices.append(key & vertex_mask)
    walk_indices.reverse()

    weight = {(u, v): w for u, neighbours in enumerate(adjacency) for v, w, _ in neighbours}
    distinct = {tuple(sorted(step)) for step in zip(walk_indices, walk_indices[1:])}
    edges = tuple(sorted((names[u], names[v]) for u, v in distinct))
    cost = sum(weight[edge] for edge in distinct)
    return MultiDemandResult(
        walk=tuple(names[i] for i in walk_indices),
        edges=edges,
        cost=cost,
        guiding_cost=guiding_cost,
        states_expanded=expanded,
    )
