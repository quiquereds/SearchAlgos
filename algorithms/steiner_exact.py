from collections.abc import Iterable, Sequence

import numpy as np

# 3^(k-1) pares de subconjuntos: con 16 terminales el cálculo tarda unos segundos.
MAX_TERMINALS = 16


def steiner_tree_cost(
    nodes: Sequence[str],
    edges: Iterable[tuple[str, str, float]],
    terminals: Iterable[str],
) -> float:
    """Costo óptimo c* del árbol de Steiner por Dreyfus-Wagner, O(3^k n + 2^k n^2).

    Sirve como referencia exacta para medir cuánto se aleja el costo reportado del óptimo.
    """
    terminals = list(dict.fromkeys(terminals))
    if len(terminals) <= 1:
        return 0.0
    if len(terminals) > MAX_TERMINALS:
        raise ValueError(f"{len(terminals)} terminales exceden el máximo de {MAX_TERMINALS}")

    index = {name: i for i, name in enumerate(nodes)}
    size = len(index)
    dist = np.full((size, size), np.inf)
    np.fill_diagonal(dist, 0.0)
    for a, b, weight in edges:
        i, j = index[a], index[b]
        dist[i, j] = dist[j, i] = min(dist[i, j], weight)
    for k in range(size):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])

    # El último terminal hace de raíz; dp[S][v] es el árbol más barato que une S con v.
    root = index[terminals[-1]]
    sources = [index[name] for name in terminals[:-1]]
    dp = np.full((1 << len(sources), size), np.inf)
    for bit, source in enumerate(sources):
        dp[1 << bit] = dist[source]

    for mask in range(1, 1 << len(sources)):
        if mask & (mask - 1) == 0:
            continue
        lowest = mask & -mask
        merged = np.full(size, np.inf)
        part = (mask - 1) & mask
        while part:
            # Exigir el bit más bajo en una de las partes evita contar cada partición dos veces.
            if part & lowest:
                np.minimum(merged, dp[part] + dp[mask ^ part], out=merged)
            part = (part - 1) & mask
        dp[mask] = (merged[:, None] + dist).min(axis=0)

    return float((dp[-1] + dist[:, root]).min())
