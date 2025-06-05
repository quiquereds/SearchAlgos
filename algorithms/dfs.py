from components.node import Node
from components.problem import Problem
from components.utils import get_root_node, print_path


def dfs(problem: Problem, explored_steps=None) -> Node | None:
    """
    Realiza búsqueda basada en profundidad (DFS) para encontrar un único estado objetivo.
    Si explored_steps se proporciona, almacena los pasos de exploración (pares padre-hijo).
    
    1. Inicializa el nodo raíz
    2. Marca el nodo actual como explorado
    3. Por cada hijo no explorado:
        a) Si es objetivo, retorna hacia la raíz,
        b) Si no es objetivo, la función se vuelve a llamar (iteración)
    
    Args:
        problem (Problem): Instancia del problema a resolver con su estado inicial y objetivo
    
    Returns:
        node? (Node): Nodo objetivo con ruta enlazada si se encuentra; de lo contrario None (no hubo solución)
    """
    
    # Inicializa el nodo raíz
    root = get_root_node(problem)
    
    # Comprobamos si el nodo actual es objetivo
    if problem.is_target(root.state):
        #*print_path(root) -> Quitar comentario para mostrar impresion en consola
        return root
    
    # Definimos un conjunto de elementos únicos para evitar repetir estados
    explored: set[str] = set()
    
    def _dfs(node: Node) -> Node | None:
        # Marcamos el nodo actual como explorado
        explored.add(node.state.name)
        # Generamos a los nodos hijos
        for child in node.expand(problem):
            if child.state.name in explored:
                continue # Saltamos si el hijo ya fue explorado
            if explored_steps is not None:
                explored_steps.append((node.state.name, child.state.name))
            if problem.is_target(child.state):
                #*print_path(root) -> Quitar comentario para mostrar impresion en consola
                return child # Si el hijo es objetivo, devolvemos la solución
            
            # Caso contrario, profundizamos en el grafo de forma recursiva
            result = _dfs(child)
            if result:
                return result
            
        # Si ninguna rama incluye al nodo objetivo devolvemos None
        print("No se encontró una solución por DFS")
        return None
    result = _dfs(root)
    if not result:
        print("No se encontró una solución por DFS")
    return result
