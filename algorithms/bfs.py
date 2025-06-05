from components.utils import get_root_node, print_path
from components.problem import Problem
from components.node import Node

def bfs(problem: Problem, explored_steps=None) -> Node | None:
    """
    Realiza búsqueda basada en anchura (BFS) para encontrar un único estado objetivo.
    En caso de que se le proporcionen varios estados objetivo, la función se detendrá al primero que encuentre
    Si explored_steps se proporciona, almacena los pasos de exploración (pares padre-hijo).
    
    1. Inicializa la cola (frontier) con la raíz del grafo
    2. Mientras existan nodos en la cola:
        a) Sacar el primer nodo,
        b) Si es objetivo, imprimir y retornar hacia la raíz,
        c) Si no es objetivo, marca como explorado y mete a la cola a los hijos
    
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
        return(root)
    
    frontier: list[Node] = [root]   # Cola FIFO
    explored: set[str] = set()      # Conjunto de estados visitados
    
    while frontier:
        node = frontier.pop(0)          # Extraemos de la cola
        explored.add(node.state.name)   # Marcamos el nodo como explorado
        
        # Generamos hijos
        for child in node.expand(problem):
            if child.state.name in explored:
                continue # Saltamos si ya fue explorado
            if problem.is_target(child.state):
                #*print_path(child) -> Quitar comentario para mostrar impresion en consola
                return child        # Si se encuentra el objetivo, devolvemos la solución
            frontier.append(child)  # Caso contrario, metemos al hijo a la cola
    
    # Si ya no hay nodos en la cola, devolvemos None
    print("No se encontró una solución por BFS")
    return None