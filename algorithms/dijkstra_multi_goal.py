from heapq import heappush, heappop
from itertools import count
from components.node import Node
from components.utils import calculate_cable_cost


class _PathLink:
    """Nodo de lista enlazada (puntero al padre) para reconstruir la ruta sin
    copiar la lista completa en cada heappush."""
    __slots__ = ("state", "parent")

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


def dijkstra_multi_goal(problem, explored_steps=None, progress_callback=None):
    """
    Algoritmo Dijkstra Multi-Objetivo
    ---------------------------------
    Encuentra el camino de costo mínimo que parte de un nodo inicial y visita todos los nodos objetivo exactamente una vez (problema tipo TSP reducido).

    - El estado de búsqueda es una tupla (nodo_actual, objetivos_restantes), donde objetivos_restantes se representa como una máscara de bits (int) de los objetivos que faltan por visitar.
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
    # Crear un diccionario de nombre a State para reconstrucción
    nodes_dict = {s.name: s for s in [problem.initial] + [t for t in problem.targets]}
    for src, adjs in problem.actions.items():
        for a, t in adjs.items():
            nodes_dict[t.name] = t

    # Cada objetivo se representa como un bit en una máscara entera en vez de
    # un frozenset de State: mismo significado, mucha menos memoria por estado.
    target_bit = {t.name: i for i, t in enumerate(problem.targets)}
    full_mask = (1 << len(target_bit)) - 1

    heap = []
    counter = count()
    heappush(heap, (0, next(counter), start.name, full_mask, _PathLink(start, None)))
    visited = {}
    nodes_explored = 0

    while heap:
        cost, _, node_name, goals_mask, link = heappop(heap)
        state = (node_name, goals_mask)
        if state in visited and visited[state] <= cost:
            continue
        visited[state] = cost
        nodes_explored += 1

        # Callback para progreso en tiempo real
        if progress_callback and nodes_explored % 10 == 0:
            progress_callback(nodes_explored, cost)

        bit = target_bit.get(node_name)
        if bit is not None:
            goals_mask &= ~(1 << bit)

        if goals_mask == 0:
            # Aplanar la cadena de punteros al padre (meta -> inicio) e invertirla
            path = []
            cur = link
            while cur is not None:
                path.append(cur.state)
                cur = cur.parent
            path.reverse()
            path_names = [s.name for s in path]

            if explored_steps is not None:
                explored_steps.append((node_name, path_names, cost))

            # Reconstruir la cadena de nodos como una lista enlazada de Node para compatibilidad
            prev = None
            total_cost = 0
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

        if explored_steps is not None and link.parent is not None:
            # Guardar el paso de llegada como par de nombres de estado para visualización
            explored_steps.append((link.parent.state.name, node_name))

        for action, neighbor in problem.actions[node_name].items():
            step_cost = problem.costs[node_name][action]
            new_cost = cost + step_cost
            new_state = (neighbor.name, goals_mask)
            # Poda antes de insertar: si ya existe un estado finalizado igual o
            # mejor, esta entrada sería descartada al sacarla del heap de todas
            # formas, así que no vale la pena ocupar memoria con ella.
            if new_state in visited and visited[new_state] <= new_cost:
                continue
            heappush(heap, (new_cost, next(counter), neighbor.name, goals_mask, _PathLink(neighbor, link)))
            # Guardar los pasos como pares de nombres de estado para visualización
            if explored_steps is not None:
                explored_steps.append((node_name, neighbor.name))
    return None
