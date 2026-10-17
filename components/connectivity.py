from collections import deque

from components.problem import Problem


class DisconnectedGraphError(ValueError):
    """El grafo tiene nodos inalcanzables desde el estado inicial."""


def reachable_from(problem: Problem, start: str) -> set[str]:
    seen = {start}
    queue = deque([start])
    while queue:
        name = queue.popleft()
        for neighbour in problem.actions.get(name, {}).values():
            if neighbour.name not in seen:
                seen.add(neighbour.name)
                queue.append(neighbour.name)
    return seen


def connected_components(problem: Problem) -> list[set[str]]:
    """Componentes conexas; asume acciones simétricas, como en las redes de conductos."""
    remaining = set(problem.actions)
    components = []
    while remaining:
        component = reachable_from(problem, next(iter(remaining)))
        components.append(component)
        remaining -= component
    return components


def unreachable_targets(problem: Problem) -> set[str]:
    reachable = reachable_from(problem, problem.initial.name)
    return {target.name for target in problem.targets} - reachable


def assert_fully_connected(problem: Problem) -> None:
    """Falla si hay nodos sueltos o demandas que el tablero no puede alcanzar."""
    components = connected_components(problem)
    if len(components) > 1:
        sizes = sorted((len(c) for c in components), reverse=True)
        stray = sorted(components[-1], key=_numeric_first)
        raise DisconnectedGraphError(
            f"el grafo tiene {len(components)} componentes (tamaños {sizes}); "
            f"nodos fuera de la principal, por ejemplo: {stray[:10]}"
        )
    missing = unreachable_targets(problem)
    if missing:
        raise DisconnectedGraphError(f"demandas inalcanzables desde {problem.initial.name}: {sorted(missing)}")


def _numeric_first(name: str) -> tuple[int, str]:
    return (0, f"{int(name):06d}") if name.isdigit() else (1, name)
