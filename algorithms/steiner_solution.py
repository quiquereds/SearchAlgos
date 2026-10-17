from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SteinerSolution:
    """Subgrafo conexo que une las terminales; edges lista cada conducto una sola vez."""

    edges: tuple[tuple[str, str], ...]
    cost: float


def build_solution(chosen: Iterable[tuple[str, str]], weights: dict[frozenset[str], float]) -> SteinerSolution:
    unique = {frozenset(edge) for edge in chosen}
    ordered = (sorted(edge, key=_node_key) for edge in unique)
    edges = tuple(sorted((a, b) for a, b in ordered))
    return SteinerSolution(edges=edges, cost=sum(weights[edge] for edge in unique))


def connects(solution: SteinerSolution, terminals: Iterable[str]) -> bool:
    """True si las aristas de la solución forman un subgrafo conexo que contiene todas las terminales."""
    terminals = set(terminals)
    if len(terminals) <= 1:
        return True
    adjacency: dict[str, set[str]] = {}
    for a, b in solution.edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    if not terminals <= adjacency.keys():
        return False
    start = next(iter(terminals))
    seen, stack = {start}, [start]
    while stack:
        for neighbour in adjacency[stack.pop()] - seen:
            seen.add(neighbour)
            stack.append(neighbour)
    return seen == adjacency.keys()


def edge_weights(edges: Iterable[tuple[str, str, float]]) -> dict[frozenset[str], float]:
    return {frozenset((a, b)): weight for a, b, weight in edges}


def _node_key(name: str) -> tuple[int, str]:
    return (0, f"{int(name):06d}") if name.isdigit() else (1, name)
