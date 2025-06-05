from ast import List
from typing import Dict, FrozenSet, Optional
from components.problem import Problem
from components.state import State
from components.action import Action

class Node:
    def __init__(
        self,
        state: State,
        parent: Optional['Node'] = None,
        action: Optional[Action] = None,
        cost: float = 0.0,
        visited: Optional[FrozenSet[str]] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['Node'] = []
        self.cost = cost
        
        # Heuristica para el nodo
        self.heuristic: Dict[str, float] = {}
        
        # Conjunto de nodos visitados
        if visited is not None:
            # 
            self.visited = visited
        else:
            self.visited = frozenset({state.name} if parent is None and False else set())
            
    def __str__(self):
        return f"Nodo: ({self.state.name}, costo: {self.cost})"
    
    def expand(self, problem: Problem) -> list['Node']:
        """
        Genera nodos hijos aplicando todas las acciones válidas.
        Actualiza el costo, los nodos visitados y la función heurística
        
        Args:
            problem (Problem): 
        Returns:
            List['Node']:
        """
        self.children = []
        for action_name, target_state in problem.actions.get(self.state.name, {}).items():
            action = Action(action_name)
            new_state = target_state
            
            # Coste acumulado
            new_cost = self.cost + problem.action_cost(self.state, action)
            
            # Determinar si el nuevo estado ya fue visitado
            new_visited = set(self.visited)
            if problem.is_target(new_state):
                new_visited.add(new_state.name)
            child = Node(
                state=new_state,
                parent=self,
                action=action,
                cost=new_cost,
                visited=frozenset(new_visited),
            )
            
            # Aplicamos la función heurística
            child.heuristic = problem.heuristic.get(child.state.name, {})
            self.children.append(child)
        return self.children