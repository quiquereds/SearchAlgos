from components.node import Node
from components.problem import Problem
from components.utils import get_root_node, print_path, calculate_cable_cost


def multi_ucs(problem: Problem, explored_steps=None) -> Node | None:
    """
    Realiza búsqueda por coste uniforme (UCS) para encontrar un trayecto que visite todos los estados objetivo.
    Si explored_steps se proporciona, almacena los pasos de exploración (pares padre-hijo).
    
    Args:
        problem (Problem): Instancia del problema a resolver con su estado inicial y objetivo
    
    Returns:
        node? (Node): Nodo objetivo con ruta enlazada si se encuentra; de lo contrario None (no hubo solución)
    """
    
    # Generamos el nodo raíz
    root = get_root_node(problem)
    
    # Obtenemos el conjunto de estados objetivo
    targets = {t.name for t in problem.targets}
    
    # Verificamos si se han explorado todos los objetivos
    if root.visited == targets:
        #print_path(root)
        return root
    
    # Creamos la frontera
    frontier: list[Node] = [root]
    # Estados explorados: Se hace un mapa (estado, visited) -> mejor coste g
    explored: dict[tuple[str, frozenset[str]], float] = {}
    
    while frontier:
        # Escogemos al nodo con menor coste
        frontier.sort(key=lambda n: n.cost)
        node= frontier.pop(0) # Extraemos al primer nodo de la cola
        key = (node.state.name, node.visited)
        
        # Saltamos si tenemos un coste igual o mejor para el par
        if explored.get(key, float("inf")) <= node.cost:
            continue
        explored[key] = node.cost
        
        # Si ya cubrimos todos los objetivos se retorna la solución
        if node.visited == targets:
            # Calcular y actualizar el costo real de cables únicos
            node.cost = calculate_cable_cost(node, problem)
            return node
        
        # Generamos a los nodos hijo
        for child in node.expand(problem):
            child_key = (child.state.name, child.visited)
            # Eliminamos si explored tiene coste <= child.cost
            if explored.get(child_key, float("inf")) <= child.cost:
                continue
            
            # Buscamos en la frontera si existe un nodo con la misma clave
            existing = next((n for n in frontier if (n.state.name, n.visited) == child_key), None)
            if existing:
                # Reemplazamos si se encuentra uno con coste menor
                if child.cost < existing.cost:
                    frontier.remove(existing)
                else:
                    continue
            if explored_steps is not None:
                explored_steps.append((node.state.name, child.state.name))
            # Agregamos el nodo a la frontera
            frontier.append(child)
    
    # Devolvemos none si la frontera se queda vacía
    print("No se encontró una solución para Multi UCS")
    return None