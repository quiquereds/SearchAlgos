from components.utils import get_root_node, print_path
from components.problem import Problem
from components.node import Node

def ucs(problem: Problem, explored_steps=None) -> Node | None:
    """
    Realiza búsqueda por coste uniforme (UCS) para encontrar un único estado objetivo minimizando el costo.
    En caso de que se le proporcionen varios estados objetivo, la función se detendrá al primero que encuentre
    
    1. Inicializa el nodo raíz con coste igual a 0
    2. Mientras la frontera no esté vacía:
        a) Se ordena frontier por coste y se extrae el nodo con coste mínimo,
        b) Si es objetivo, se devuelve la solución,
        c) Si no es objetivo, se marca como explorado y se generan a los hijos:
            - Si un hijo aparece con menor costo, se reemplaza por el de mayor costo
    
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
    
    frontier: list[Node] = [root] # Cola
    explored: set[str] = set() # Conjunto para estados visitados
    
    while frontier:
        # Ordenamos la frontera por coste acumulado
        frontier.sort(key= lambda n: n.cost)
        node = frontier.pop(0) # Extraemos el primer nodo de la cola
        
        # Comprobamos si el nodo es objetivo
        if problem.is_target(node.state):
            #*print_path(root) -> Quitar comentario para mostrar impresion en consola
            return node

        # De lo contrario, marcamos como explorado y generamos a los hijos
        explored.add(node.state.name)
        for child in node.expand(problem):
            if child.state.name in explored:
                continue
            # Buscamos en la frontera algun nodo con el mismo estado
            existing = next((n for n in frontier if n.state.name == child.state.name), None)
            if existing:
                # Reemplazamos si encontramos uno con coste menor
                if child.cost < existing.cost:
                    frontier.remove(existing)
                else:
                    continue
            if explored_steps is not None:
                explored_steps.append((node.state.name, child.state.name))
            frontier.append(child)
    
    print("No se encontró una solución para UCS")
    return None