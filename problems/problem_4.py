
from algorithms.astar import astar
from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.dijkstra import dijkstra
from algorithms.dijkstra_multi_goal import dijkstra_multi_goal
from algorithms.ucs import ucs
from algorithms.ucs_multi_goal import multi_ucs
from benchmark import print_benchmark_table, run_benchmark
from components.problem import Problem
from components.state import State
from components.visualization import run_and_animate_nx

# 1) Definimos todos los nodos de 0 a 37
nodes = { str(i): State(str(i)) for i in range(0, 38) }

# 2) Definimos las conexiones entre nodos y sus costos
edges = [
    ('0', '1', 2),
    ('0', '20', 3.8),
    ('1', '7', 3.8),
    ('1', '2', 2.8),
    ('2', '8', 4),
    ('3', '9', 3.65),
    ('4', '10', 3.4),
    ('5', '11', 3.55),
    ('6', '8', 2.5),
    ('7', '15', 3.7),
    ('8', '9', 2.1),
    ('8', '15', 1.5),
    ('9', '3', 3.65),
    ('9', '10', 2.3),
    ('9', '16', 3.5),
    ('9', '22', 2.2),
    ('10', '4', 3.4),
    ('10', '11', 2.4),
    ('10', '17', 3.65),
    ('11', '5', 3.55),
    ('11', '14', 3.2),
    ('11', '18', 3.6),
    ('11', '23', 4.3),
    ('12', '23', 2.5),
    ('13', '23', 2.8),
    ('14', '19', 3.45),
    ('15', '22', 2.8),
    ('16', '9', 3.5),
    ('17', '10', 3.65),
    ('18', '11', 3.6),
    ('19', '25', 4.2),
    ('20', '21', 2.85),
    ('21', '22', 3.6),
    ('21', '33', 5.2),
    ('22', '27', 3.85),
    ('22', '32', 6),
    ('22', '30', 4.5),
    ('23', '24', 3.1),
    ('23', '28', 3.85),
    ('25', '29', 2),
    ('26', '30', 3.6),
    ('28', '32', 3.6),
    ('29', '32', 3),
    ('30', '31', 3),
    ('30', '36', 3.5),
    ('31', '32', 3.2),
    ('31', '34', 2.5),
    ('32', '37', 4.1),
    ('32', '34', 3.5),
    ('33', '35', 2),
    ('35', '36', 3),
]

# 3) Construimos acciones y costos
actions = { name: {} for name in nodes }
costs   = { name: {} for name in nodes }

for u, v, w in edges:
    actions[u][f"to{v}"] = nodes[v]
    costs[u][f"to{v}"]   = w
    actions[v][f"to{u}"] = nodes[u]
    costs[v][f"to{u}"]   = w

def create_heuristic(nodes, target_nodes):
    heuristic = {}
    for node_name in nodes:
        heuristic[node_name] = {}
        for target_name in target_nodes:
            node_num = int(node_name)
            target_num = int(target_name)
            heuristic[node_name][target_name] = abs(node_num - target_num)
    return heuristic

target_nodes = ['20']
heuristic = create_heuristic(nodes, target_nodes)


# 4) Definimos los problemas de único y múltiple objetivo
prob_single = Problem(
    initial=nodes['0'],  # nodo inicial
    targets=[nodes['20']],  # nodo objetivo
    actions=actions,
    costs=costs,
    heuristic=heuristic
)

prob_multi = Problem(
    initial=nodes['0'],
    targets=[nodes['0'], nodes['1'], nodes['7'], nodes['15'], nodes['22'], nodes['23'], nodes['24']],
    actions=actions,
    costs=costs,
    heuristic=create_heuristic(nodes, ['0', '1', '7', '15', '22', '23', '24'])
)

if __name__ == "__main__":
    multi_goal = {
        'A*': astar,
        'Multi-UCS': multi_ucs,
        'Dijkstra-MultiGoal': dijkstra_multi_goal
    }
    single_goal = {
        'BFS': bfs,
        'DFS': dfs,
        'UCS': ucs,
        'Dijkstra': dijkstra,
        'A*': astar
    }
    
    # Problema con objetivo único (del nodo 0 al 20)
    print("=== PROBLEMA CON OBJETIVO ÚNICO ===")
    results_single = run_benchmark(prob_single, single_goal)
    for r in results_single:
        r['Caso'] = 'Objetivo único'
    print_benchmark_table(results_single)
    
    # Problema con múltiples objetivos
    print("\n=== PROBLEMA CON MÚLTIPLES OBJETIVOS ===")
    results_multi = run_benchmark(prob_multi, multi_goal)
    for r in results_multi:
        r['Caso'] = 'Multi-objetivo'
    print_benchmark_table(results_multi)

    print("\nAnimación del proceso de búsqueda en curso:")
    run_and_animate_nx(prob_multi, astar, nodes, actions, costs, speed='muy_rapido')

