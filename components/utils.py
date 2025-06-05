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

def calculate_cable_cost(node: Node, problem: Problem) -> float:
    """
    Calcula el costo real de cables únicos utilizados en la ruta.
    Solo cuenta cada arista una vez, sin importar cuántas veces se use.
    
    Args:
        node (Node): Nodo final con la ruta completa
        problem (Problem): Problema con los costos de las acciones
        
    Returns:
        float: Costo total de cables únicos utilizados
    """
    path = reconstruct_path(node)
    used_edges = set()  # Para almacenar aristas únicas
    total_cable_cost = 0.0
    
    # Recorrer la ruta y contar cada arista única
    for i in range(len(path) - 1):
        current_state = path[i]
        next_state = path[i + 1]
        
        # Crear una representación única de la arista (sin dirección)
        edge = tuple(sorted([current_state, next_state]))
        
        if edge not in used_edges:
            used_edges.add(edge)
            # Buscar el costo de esta arista en las acciones del problema
            edge_cost = None
            for action_name, target_state in problem.actions.get(current_state, {}).items():
                if target_state.name == next_state:
                    edge_cost = problem.costs[current_state][action_name]
                    break
            
            if edge_cost is not None:
                total_cable_cost += edge_cost
            else:
                print(f"Warning: No se encontró costo para la arista {current_state} -> {next_state}")
    
    return total_cable_cost

def print_path_with_cable_cost(node: Node, problem: Problem):
    """
    Muestra la ruta, el coste algorítmico y el coste real de cables
    """
    path = reconstruct_path(node)
    cable_cost = calculate_cable_cost(node, problem)
    print("Ruta: " + " -> ".join(path))
    print(f"Coste algorítmico: {node.cost}")
    print(f"Coste real de cables: {cable_cost}")
    print(f"Ahorro de cable: {node.cost - cable_cost:.2f}")