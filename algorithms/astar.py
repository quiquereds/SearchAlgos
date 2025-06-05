from components.utils import get_root_node, print_path, calculate_cable_cost
from components.problem import Problem
from components.node import Node
import heapq
import itertools

def astar(problem: Problem, explored_steps=None) -> Node | None:
    """
    Combina coste acumulado g(n) y heurística mínima hacia objetivos restantes h(n), expandiendo nodos con menor
    f(n) = g + H.
    
    Se utiliza un heap que es una estructura de datos que permite almacenar y acceder a elementos de forma eficiente,
    permitiendo definir colas de prioridad, de esta forma, cada elemento tiene asignada una prioridad y los elementos
    con mayor prioridad se colocan en la parte superior. En este caso, extraeremos elementos con la prioridad más baja
    de f = g + h.
    
    En resumen, el algoritmo realiza lo siguiente:
    
    1. Construye una función heurística que considera solo a los objetivos no visitados
    2. Usa un heap (frontera) de tuplas (f, contador, nodo)
    3. Elimina estados visitados por mejor g(n)
    4. Se detiene al cubrir todos los objetivos
    
    Args:
        problem (Problem): Instancia del problema a resolver con su estado inicial y objetivo
    
    Returns:
        node? (Node): Nodo objetivo con ruta enlazada si se encuentra; de lo contrario None (no hubo solución)
    """
    
    # Inicializamos el nodo raíz con g = 0
    root = get_root_node(problem)
    targets = {t.name for t in problem.targets} # Conjunto de objetivos por nombre
    
    # Función heurísitica con la mínima estimación a objetivos no visitados
    def heuristic(node: Node) -> float:
        remaining = targets - node.visited # Determinados objetivos que aún no han sido cubiertos
        if not remaining: 
            return 0.0 # Si se han cubrido todos los objetivos, devolvemos h = 0
        # Caso contrario, retornamos el valor heurístico entre estados restantes
        return min(node.heuristic.get(t, float('inf')) for t in remaining)

    # Definimos la frontera como una cola de prioridad
    frontier: list[tuple[float, int, Node]] = []  # Cada tupla está definida como (f, contador, nodo)
    counter = itertools.count() # Creamos un contador que aumentará cada que se insrte un nodo y asi el algoritmo determine que nodo sale primero

    # Insertamos el nodo raíz a la cola cuya prioridad se calculará en base a f = g + h(root)
    heapq.heappush(frontier, (root.cost + heuristic(root), next(counter), root))

    # Creamos un diccionario que guardará el menor coste g(n) para cada combinación (estado, visited)
    best_g: dict[tuple[str, frozenset[str]], float] = {
        (root.state.name, root.visited): 0.0
    }

    while frontier:
        f, _, node = heapq.heappop(frontier)        # Extraemos el nodo con menor f
        g = node.cost                               # Obtenemos el coste acumulado al nodo
        key = (node.state.name, node.visited)
        
        # Realizamos una comprobación para ver si se cubrieron todos los objetivod
        if node.visited == targets:
            # Calcular y actualizar el costo real de cables únicos
            node.cost = calculate_cable_cost(node, problem)
            return node # Si se encontró solución, la devolvemos
        
        # Eliminamos el nodo si el coste g ya no es óptimo
        if g > best_g.get(key, float('inf')):
            continue
        
        # Expandimos a los nodos hijos
        for child in node.expand(problem):
            child_key = (child.state.name, child.visited)
            g2 = child.cost             # Nuevo coste g
            h2 = heuristic(child)       # Nueva prioridad f
            f2 = g2 + h2
            
            # Si encontramos un mejor g2, lo registramos y lo añadimos a la cola
            if g2 < best_g.get(child_key, float('inf')):
                best_g[child_key] = g2
                if explored_steps is not None:
                    explored_steps.append((node.state.name, child.state.name))
                heapq.heappush(frontier, (f2, next(counter), child))
    
    # Si se queda vacía la cola sin encontrar solución, devolvemos none
    print("No se encontró una solución para A*")
    return None