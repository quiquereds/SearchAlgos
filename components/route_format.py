import textwrap
from collections.abc import Iterable, Mapping


def format_walk(walk: Iterable[str], width: int = 100) -> str:
    """Recorrido de la búsqueda como 0 → 1 → 7, partido en líneas de ancho razonable."""
    return textwrap.fill(" → ".join(walk), width=width, break_long_words=False, break_on_hyphens=False)


def format_tree(
    edges: Iterable[tuple[str, str]],
    root: str,
    weights: Mapping[frozenset[str], float],
    demand: Iterable[str] = (),
) -> str:
    """Árbol de cable dibujado desde el tablero; las demandas llevan * y cada rama su longitud.

    Si las aristas contienen ciclos, las que no caben en el árbol se listan al final.
    """
    edges = [(a, b) for a, b in edges]
    demand = set(demand)
    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    children: dict[str, list[str]] = {}
    seen = {root}

    def build(node: str) -> None:
        for neighbour in sorted(adjacency.get(node, ()), key=_node_key):
            if neighbour not in seen:
                seen.add(neighbour)
                children.setdefault(node, []).append(neighbour)
                build(neighbour)

    build(root)

    def label(name: str) -> str:
        return f"{name}*" if name in demand else name

    lines = [label(root)]

    def render(node: str, prefix: str) -> None:
        kids = children.get(node, [])
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1
            length = weights[frozenset((node, kid))]
            lines.append(f"{prefix}{'└─' if last else '├─'} {label(kid)} ({length:g} m)")
            render(kid, prefix + ("   " if last else "│  "))

    render(root, "")

    in_tree = {frozenset((parent, kid)) for parent, kids in children.items() for kid in kids}
    extra = sorted({frozenset(edge) for edge in edges} - in_tree, key=lambda e: sorted(e, key=_node_key))
    if extra:
        names = ", ".join("–".join(sorted(e, key=_node_key)) for e in extra)
        lines.append(f"Aristas que cierran ciclos: {names}")
    return "\n".join(lines)


def _node_key(name: str) -> tuple[int, int, str]:
    return (0, int(name), "") if name.isdigit() else (1, 0, name)
