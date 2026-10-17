from collections.abc import Iterable, Sequence
from heapq import heappop, heappush

from algorithms.steiner_solution import SteinerSolution, build_solution, edge_weights


def takahashi_matsuyama(
    nodes: Sequence[str],
    edges: Iterable[tuple[str, str, float]],
    root: str,
    terminals: Iterable[str],
) -> SteinerSolution | None:
    """Heurística de Takahashi y Matsuyama (1980) para el árbol de Steiner.

    Parte del árbol {root} y, en cada paso, conecta la terminal más cercana al árbol actual
    por un camino mínimo. Es 2-aproximada y corre en tiempo polinomial.
    Devuelve None si alguna terminal es inalcanzable desde root.
    """
    edges = list(edges)
    weights = edge_weights(edges)
    index = {name: i for i, name in enumerate(nodes)}
    adjacency: dict[str, list[tuple[str, float]]] = {name: [] for name in nodes}
    for a, b, weight in edges:
        adjacency[a].append((b, weight))
        adjacency[b].append((a, weight))

    in_tree = {root}
    pending = set(terminals) - in_tree
    chosen: list[tuple[str, str]] = []

    while pending:
        # Dijkstra multi-origen desde todo el árbol; el desempate por índice lo hace determinista.
        dist = {name: 0.0 for name in in_tree}
        parent: dict[str, str] = {}
        heap = [(0.0, index[name], name) for name in in_tree]
        reached = None
        while heap:
            d, _, name = heappop(heap)
            if d > dist[name]:
                continue
            if name in pending:
                reached = name
                break
            for neighbour, weight in adjacency[name]:
                new_d = d + weight
                if new_d < dist.get(neighbour, float("inf")):
                    dist[neighbour] = new_d
                    parent[neighbour] = name
                    heappush(heap, (new_d, index[neighbour], neighbour))
        if reached is None:
            return None

        name = reached
        while name not in in_tree:
            chosen.append((parent[name], name))
            in_tree.add(name)
            name = parent[name]
        pending -= in_tree

    return build_solution(chosen, weights)
