from components.node import Node
from components.problem import Problem
from components.utils import get_root_node, print_path
import heapq
import itertools

def dijkstra(problem: Problem, explored_steps=None) -> Node | None:
    """
    Realiza la búsqueda de caminos mínimos usando el algoritmo de Dijkstra para encontrar el estado objetivo
    con el menor coste acumulado desde el estado inicial. Dijkstra es equivalente a UCS cuando todos los costes
    son positivos y no utiliza heurística.

    1. Inicializa el nodo raíz con coste igual a 0
    2. Utiliza una cola de prioridad (heap) para seleccionar el nodo con menor coste acumulado
    3. Marca los estados explorados y actualiza el coste mínimo para cada estado
    4. Se detiene al encontrar el primer estado objetivo

    Args:
        problem (Problem): Instancia del problema a resolver con su estado inicial y objetivo

    Returns:
        node? (Node): Nodo objetivo con ruta enlazada si se encuentra; de lo contrario None (no hubo solución)
    """
    
    # Inicializamos el nodo raíz
    root = get_root_node(problem)
    
    # Comprobamos si el nodo actual es objetivo
    if problem.is_target(root.state):
        #*print_path(root) -> Quitar comentario para mostrar impresion en consola
        return root
    
    # Definimos la frontera como un heap de tuplas (coste, contador, nodo)
    frontier: list[tuple[float, int, Node]] = []
    counter = itertools.count()
    heapq.heappush(frontier, (root.cost, next(counter), root))
    
    # Diccionario para registrar el menor coste a cada estado
    best_g: dict[str, float] = {root.state.name: 0.0}
    
    while frontier:
        cost, _, node = heapq.heappop(frontier)
        
        # Si el nodo es objetivo, devolvemos la solución
        if problem.is_target(node.state):
            #*print_path(node) -> Quitar comentario para mostrar impresion en consola
            return node
        
        # Expandimos a los hijos
        for child in node.expand(problem):
            g2 = child.cost
            # Si encontramos un mejor coste, lo registramos y lo añadimos a la frontera
            if g2 < best_g.get(child.state.name, float('inf')):
                best_g[child.state.name] = g2
                if explored_steps is not None:
                    explored_steps.append((node.state.name, child.state.name))
                heapq.heappush(frontier, (g2, next(counter), child))
    
    print("No se encontró una solución para Dijkstra")
    return None
