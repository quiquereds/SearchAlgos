import time
from typing import Callable, Dict, List
from components.utils import reconstruct_path
from components.problem import Problem
import pandas as pd

from algorithms.bfs import bfs
from algorithms.ucs import ucs
from algorithms.astar import astar
from algorithms.ucs_multi_goal import multi_ucs
from algorithms.dfs import dfs
from algorithms.dijkstra import dijkstra


def run_benchmark(
    problem: Problem,
    algos: Dict[str, Callable[[Problem], object]]
) -> List[Dict]:
    """
    Ejecuta un conjunto de algoritmos de búsqueda sobre un problema y recopila métricas de desempeño.
    Args:
        problem (Problem): Instancia del problema a resolver.
        algos (Dict[str, Callable]): Diccionario de algoritmos a ejecutar.
    Returns:
        List[Dict]: Resultados con nombre, tiempo, costo y ruta de cada algoritmo.
    """
    results = []
    for name, fn in algos.items():
        start = time.perf_counter()
        node = fn(problem)
        elapsed = time.perf_counter() - start

        if node:
            path = reconstruct_path(node)
            cost = node.cost
            route = " → ".join(path)
        else:
            cost = float('inf')
            route = ""

        results.append({
            'Algoritmo': name,
            'Tiempo (s)': round(elapsed, 6),
            'Costo': cost,
            'Ruta': route
        })
    return results


def print_benchmark_table(results: List[Dict]):
    """
    Imprime los resultados del benchmark en formato tabla ordenada.
    Args:
        results (List[Dict]): Resultados a mostrar.
    """
    df = pd.DataFrame(results)
    df = df.sort_values(by=['Caso', 'Costo', 'Tiempo (s)'])
    print(df.to_markdown(index=False, tablefmt="github"))


if __name__ == "__main__":
    from problems.problem_1 import prob_single, prob_multi

    benchmarks = [
        ('Único objetivo', prob_single),
        ('Multi objetivo', prob_multi)
    ]
    algos = {
        'BFS': bfs,
        'DFS': dfs,
        'UCS': ucs,
        'Dijkstra': dijkstra,
        'A*': astar,
        'Multi-UCS': multi_ucs
    }

    all_results = []
    for label, prob in benchmarks:
        res = run_benchmark(prob, algos)
        for r in res:
            r['Caso'] = label
        all_results.extend(res)

    print_benchmark_table(all_results)