from components.state import State
from components.problem import Problem
from benchmark import run_benchmark, print_benchmark_table
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.ucs import ucs
from algorithms.dijkstra import dijkstra
from algorithms.astar import astar
from algorithms.ucs_multi_goal import multi_ucs
from algorithms.dijkstra_multi_goal import dijkstra_multi_goal
from components.visualization import plot_graph_and_path_nx, run_and_animate_nx
import plotly.graph_objects as go

# Definición de un problema de grafo grande y complejo
# 20 nodos, conexiones densas, costos variados
nodes = {chr(65+i): State(chr(65+i)) for i in range(20)}  # Nodos A-T

# Acciones
actions = {
    'A': {'toB': nodes['B'], 'toC': nodes['C'], 'toD': nodes['D'], 'toE': nodes['E']},
    'B': {'toF': nodes['F'], 'toG': nodes['G'], 'toH': nodes['H']},
    'C': {'toI': nodes['I'], 'toJ': nodes['J'], 'toK': nodes['K']},
    'D': {'toL': nodes['L'], 'toM': nodes['M']},
    'E': {'toN': nodes['N'], 'toO': nodes['O']},
    'F': {'toP': nodes['P'], 'toQ': nodes['Q']},
    'G': {'toR': nodes['R'], 'toS': nodes['S']},
    'H': {'toT': nodes['T'], 'toA': nodes['A']},
    'I': {'toB': nodes['B'], 'toD': nodes['D']},
    'J': {'toE': nodes['E'], 'toF': nodes['F']},
    'K': {'toG': nodes['G'], 'toH': nodes['H']},
    'L': {'toI': nodes['I'], 'toJ': nodes['J']},
    'M': {'toK': nodes['K'], 'toL': nodes['L']},
    'N': {'toM': nodes['M'], 'toN': nodes['N']},
    'O': {'toO': nodes['O'], 'toP': nodes['P']},
    'P': {'toQ': nodes['Q'], 'toR': nodes['R']},
    'Q': {'toS': nodes['S'], 'toT': nodes['T']},
    'R': {'toA': nodes['A'], 'toB': nodes['B']},
    'S': {'toC': nodes['C'], 'toD': nodes['D']},
    'T': {'toE': nodes['E'], 'toF': nodes['F']},
}

# Costos 
costs = {
    'A': {'toB': 2, 'toC': 5, 'toD': 9, 'toE': 4},
    'B': {'toF': 7, 'toG': 3, 'toH': 8},
    'C': {'toI': 6, 'toJ': 2, 'toK': 4},
    'D': {'toL': 3, 'toM': 5},
    'E': {'toN': 8, 'toO': 6},
    'F': {'toP': 1, 'toQ': 7},
    'G': {'toR': 2, 'toS': 9},
    'H': {'toT': 4, 'toA': 10},
    'I': {'toB': 3, 'toD': 6},
    'J': {'toE': 2, 'toF': 5},
    'K': {'toG': 7, 'toH': 1},
    'L': {'toI': 2, 'toJ': 8},
    'M': {'toK': 4, 'toL': 6},
    'N': {'toM': 3, 'toN': 2},
    'O': {'toO': 5, 'toP': 7},
    'P': {'toQ': 2, 'toR': 3},
    'Q': {'toS': 6, 'toT': 1},
    'R': {'toA': 8, 'toB': 2},
    'S': {'toC': 4, 'toD': 9},
    'T': {'toE': 3, 'toF': 5},
}

# Heurística: distancia estimada a T
heuristic = {name: {'T': abs(ord('T') - ord(name))} for name in nodes}

# Definir problema: de A a T
prob_hard = Problem(
    initial=nodes['A'],
    targets=[nodes['T']],
    actions=actions,
    costs=costs,
    heuristic=heuristic
)

# Ejemplo de problema multiobjetivo: visitar T, P y M
prob_multi = Problem(
    initial=nodes['A'],
    targets=[nodes['M'], nodes['P'], nodes['T']],
    actions=actions,
    costs=costs,
    heuristic=heuristic
)

if __name__ == "__main__":
    algos = {
        'BFS': bfs,
        'DFS': dfs,
        'UCS': ucs,
        'Dijkstra': dijkstra,
        'A*': astar,
        'Multi-UCS': multi_ucs,
        'Dijkstra-MultiGoal': dijkstra_multi_goal
    }
    results = run_benchmark(prob_multi, algos)
    for r in results:
        r['Caso'] = 'Problema 3'
    print_benchmark_table(results)

    print("\nAnimación del proceso de búsqueda en curso:")
    run_and_animate_nx(prob_multi, dijkstra_multi_goal, nodes, actions, costs)

    dijkstra_result = next((r for r in results if r['Algoritmo'] == 'Dijkstra'), None)
    if dijkstra_result and dijkstra_result['Ruta']:
        path = [n.strip() for n in dijkstra_result['Ruta'].split('→')]
        plot_graph_and_path_nx(nodes, actions, costs, path)
    else:
        plot_graph_and_path_nx(nodes, actions, costs)
