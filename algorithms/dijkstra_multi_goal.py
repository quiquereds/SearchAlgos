from heapq import heappush, heappop
from itertools import count
from components.node import Node
from components.state import State
from components.utils import calculate_cable_cost

def dijkstra_multi_goal(problem, explored_steps=None):
    """
    Algoritmo Dijkstra Multi-Objetivo
    ---------------------------------
    Encuentra el camino de costo mínimo que parte de un nodo inicial y visita todos los nodos objetivo exactamente una vez (problema tipo TSP reducido).
    
    - El estado de búsqueda es una tupla (nodo_actual, objetivos_restantes), donde objetivos_restantes es un conjunto inmutable (frozenset) de los objetivos que faltan por visitar.
    - Utiliza una cola de prioridad (heapq) para expandir primero los caminos de menor costo acumulado.
    - Al llegar a un nodo objetivo, lo elimina del conjunto de objetivos restantes.
    - Cuando el conjunto de objetivos restantes está vacío, retorna la ruta y el costo total.
    - Registra los pasos explorados en explored_steps (si se provee) para visualización o análisis.
    
    Args:
        problem: instancia de Problem con initial, targets, actions, costs.
        explored_steps: lista opcional para registrar los pasos explorados (para animación/visualización).
    
    Returns:
        dict con las claves:
            'path': lista de nodos visitados en orden.
            'cost': costo total del recorrido.
            'explored_steps': lista de pasos explorados (si se solicitó).
    """
    start = problem.initial
    goals = set(problem.targets)
    # Crear un diccionario de nombre a State para reconstrucción
    nodes_dict = {s.name: s for s in [problem.initial] + [t for t in problem.targets]}
    for src, adjs in problem.actions.items():
        for a, t in adjs.items():
            nodes_dict[t.name] = t
    heap = []
    counter = count()
    heappush(heap, (0, next(counter), start, frozenset(goals), [start]))
    visited = {}

    while heap:
        cost, _, node, goals_left, path = heappop(heap)
        state = (node, goals_left)
        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost

        if node in goals_left:
            goals_left = goals_left - {node}
        if not goals_left:
            if explored_steps is not None:
                explored_steps.append((node, list(path), cost))
            # Reconstruir la cadena de nodos como una lista enlazada de Node para compatibilidad
            prev = None
            total_cost = 0
            path_names = [s.name for s in path]
            for i in range(len(path_names)):
                name = path_names[i]
                if i == 0:
                    n = Node(state=nodes_dict[name], parent=None, action=None, cost=0)
                else:
                    src = path_names[i-1]
                    dst = path_names[i]
                    found = False
                    for a, t in problem.actions[src].items():
                        if t.name == dst:
                            step_cost = problem.costs[src][a]
                            total_cost += step_cost
                            found = True
                            break
                    if not found:
                        raise ValueError(f"No se encontró acción de {src} a {dst} al reconstruir la ruta en dijkstra_multi_goal.")
                    n = Node(state=nodes_dict[dst], parent=prev, action=None, cost=total_cost)
                prev = n
            
            # Calcular el costo real de cables únicos
            real_cable_cost = calculate_cable_cost(prev, problem)
            prev.cost = real_cable_cost  # Actualizar con el costo real de cables
            return prev

        if explored_steps is not None:
            # Guardar los pasos como pares de nombres de estado para visualización
            if len(path) > 1:
                explored_steps.append((path[-2].name, node.name))
            # Asegurarse de que todos los pasos sean str, nunca State
            explored_steps[:] = [
                (e[0].name if isinstance(e[0], State) else e[0],
                 e[1].name if isinstance(e[1], State) else e[1])
                if isinstance(e, tuple) and len(e) == 2 else e
                for e in explored_steps
            ]

        for action, neighbor in problem.actions[node.name].items():
            step_cost = problem.costs[node.name][action]
            heappush(heap, (cost + step_cost, next(counter), neighbor, goals_left, path + [neighbor]))
            # Guardar los pasos como pares de nombres de estado para visualización
            if explored_steps is not None:
                explored_steps.append((node.name, neighbor.name))
        if not goals_left:
            if explored_steps is not None:
                # También convertir los pasos de la ruta óptima a pares de nombres
                for i in range(1, len(path)):
                    explored_steps.append((path[i-1].name, path[i].name))
    return None
