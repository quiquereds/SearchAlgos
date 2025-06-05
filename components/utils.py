from typing import List
from components.node import Node
from components.problem import Problem

def get_root_node(problem: Problem) -> Node:
    """
    Crea el nodo raíz con coste igual a 0 y su estado visitado inicial
    """
    
    root = Node(
        state=problem.initial,
        parent=None,
        action=None,
        cost=0.0,
        visited=frozenset({problem.initial.name} if problem.is_target(problem.initial) else set())
    )
    
    # Aplicar función heurística
    root.heuristic = problem.heuristic.get(root.state.name, {})
    return root

def reconstruct_path(node: Node) -> None:
    """
    Devuelve la lista de nombres de estado desde la raíz hasta el nodo solución
    """
    path: list[str] = []
    current = node
    while current:
        path.append(current.state.name)
        current = current.parent
    return list(reversed(path))

def print_path(node: Node):
    """
    Muestra la ruta y el coste total del nodo final
    """
    path = reconstruct_path(node)
    print("Ruta: " + " -> ".join(path))
    print(f"Coste total: {node.cost}")