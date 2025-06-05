
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

# 1) Definimos todos los nodos de 1 a 61
nodes = { str(i): State(str(i)) for i in range(1, 62) }

# 2) Definimos las conexiones entre nodos y sus costos
edges = [
    ('1', '6', 3.5),
    ('2', '50', 5.2),
    ('2', '7', 4.7),
    ('3', '8', 4.8),
    ('4', '9', 5.2),
    ('5', '13', 4.2),
    ('6', '50', 4.8),
    ('6', '16', 4.5),
    ('7', '50', 3.8),
    ('7', '17', 3.2),
    ('7', '8', 3.4),
    ('8', '3', 4.8),
    ('8', '18', 3.2),
    ('8', '9', 4.1),
    ('9', '4', 5.2),
    ('9', '19', 3.2),
    ('9', '10', 2.5),
    ('10', '20', 3.2),
    ('10', '21', 5.8),
    ('10', '11', 3.8),
    ('11', '12', 3.5),
    ('12', '22', 3.5),
    ('12', '13', 3.1),
    ('13', '5', 4.2),
    ('13', '23', 3.5),
    ('14', '21', 3.8),
    ('15', '16', 4.3),
    ('16', '38', 4.5),
    ('16', '29', 5.5),
    ('17', '50', 6.2),
    ('17', '24', 3.3),
    ('17', '30', 3.8),
    ('17', '18', 3.5),
    ('18', '19', 4.2),
    ('19', '25', 3.7),
    ('19', '20', 2.5),
    ('21', '22', 3.7),
    ('21', '26', 3.7),
    ('22', '27', 3.5),
    ('22', '23', 3.2),
    ('23', '28', 4.2),
    ('23', '37', 3.6),
    ('29', '30', 3.4),
    ('30', '43', 5.9),
    ('30', '31', 4.5),
    ('31', '44', 5.3),
    ('31', '32', 3.9),
    ('32', '34', 3.8),
    ('32', '33', 4.2),
    ('34', '35', 3.5),
    ('35', '46', 5.8),
    ('36', '37', 3.7),
    ('37', '48', 5.6),
    ('38', '49', 5.2),
    ('39', '43', 3.6),
    ('40', '45', 3.5),
    ('41', '48', 4.2),
    ('42', '43', 3.9),
    ('43', '50', 3.5),
    ('43', '44', 4.3),
    ('44', '51', 3.6),
    ('44', '45', 4.5),
    ('45', '46', 3.5),
    ('45', '57', 4.2),
    ('46', '58', 4.3),
    ('46', '47', 3.8),
    ('47', '59', 3.7),
    ('47', '48', 3.2),
    ('48', '60', 3.7),
    ('49', '53', 4.8),
    ('49', '50', 5.2),
    ('50', '54', 4.2),
    ('50', '51', 4.4),
    ('52', '59', 3.8),
    ('54', '55', 3.9),
    ('56', '57', 3.8),
    ('57', '58', 3.5),
    ('58', '59', 4.2),
    ('59', '60', 3.2),
    ('60', '61', 2.5),
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

target_nodes = ['60']
heuristic = create_heuristic(nodes, target_nodes)


# 4) Definimos los problemas de único y múltiple objetivo
prob_single = Problem(
    initial=nodes['1'],  # nodo inicial
    targets=[nodes['60']],  # nodo objetivo
    actions=actions,
    costs=costs,
    heuristic=heuristic
)

prob_multi = Problem(
    initial=nodes['1'],
    targets=[nodes['1'], nodes['6'], nodes['50'], nodes['2'], nodes['3'], nodes['4'], nodes['5'], nodes['14'], nodes['15'], nodes['24'], nodes['25'], nodes['26'],nodes['27'], nodes['28']],
    actions=actions,
    costs=costs,
    heuristic=create_heuristic(nodes, ['1', '6', '50', '2', '3', '4', '5', '14', '15', '24', '25', '26', '27', '28'])
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
    
    # Visualización problema con objetivo único (del nodo 0 al 20)
    print("=== PROBLEMA CON OBJETIVO ÚNICO ===")
    results_single = run_benchmark(prob_single, single_goal)
    for r in results_single:
        r['Caso'] = 'Objetivo único'
    print_benchmark_table(results_single)
    
    # Visualización problema con múltiples objetivos
    print("\n=== PROBLEMA CON MÚLTIPLES OBJETIVOS ===")
    results_multi = run_benchmark(prob_multi, multi_goal)
    for r in results_multi:
        r['Caso'] = 'Multi-objetivo'
    print_benchmark_table(results_multi)

    print("\nAnimación del proceso de búsqueda en curso:")
    run_and_animate_nx(prob_multi, multi_ucs, nodes, actions, costs, speed='muy_rapido')

