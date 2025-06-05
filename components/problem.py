from components.action import Action
from components.state import State
from typing import List, Dict

class Problem:
    """
    Clase que representa un problema de búsqueda. Contiene el estado inicial,
    los estados objetivo, las acciones y los costos asociados a cada acción.
    El problema se define por un estado inicial, una lista de estados objetivo,
    un diccionario de acciones y un diccionario de costos.
    
    El estado inicial es el punto de partida del problema, y los estados objetivo
    son los estados que se desean alcanzar. Las acciones son las posibles
    transiciones entre estados, y los costos son los valores asociados a cada acción.
    
    La heurística es un diccionario que puede ser utilizado para guiar la búsqueda
    hacia los estados objetivo. La heurística puede ser utilizada para estimar el
    costo de alcanzar un estado objetivo desde un estado dado.
    """
    def __init__(
        self,
        initial,
        targets: List[State],
        actions: Dict[str, Dict[str, State]],
        costs: Dict[str, Dict[str, float]] = None,
        heuristic: Dict[str, Dict[str, float]] = None,
    ):
        """
        Inicializa el problema con el estado inicial, los estados objetivo, las acciones y los costos.
        
        Args:
            initial (State): Estado inicial del problema.
            targets (List[State]): Lista de estados objetivo.
            actions (Dict[str, Dict[str, State]]): Diccionario de acciones y sus estados resultantes.
            costs (Dict[str, Dict[str, float]], optional): Costos de las acciones. Por defecto None.
            heuristic (Dict[str, Dict[str, float]], optional): Heurística para el problema. Por defecto None.
        """
        self.initial = initial
        self.targets = targets
        self.actions = actions
        
        # Costos por defecto a 1 si no se especifican
        self.costs = costs or {s: {a: 1.0 for a in acts} for s, acts in actions.items()}
        
        # Heurística por defecto a infinito si no se especifica
        if heuristic:
            self.heuristic = heuristic
        else:
            self.heuristic = {
                s: {t.name: (0 if s == t.name else float("inf")) for t in targets} for s in actions
            }
            
    def is_target(self, state: State) -> bool:
        """
        Verifica si un estado es uno de los estados objetivo.
        
        Args:
            state (State): Estado a verificar.
        
        Returns:
            bool: True si el estado es objetivo, False en caso contrario.
        """
        return state.name in {t.name for t in self.targets}
    
    def result(self, state: State, action: Action) -> State:
        """
        Devuelve el estado resultante de aplicar una acción a un estado dado.
        
        Args:
            state (State): Estado inicial.
            action (Action): Acción a aplicar.
        
        Returns:
            State: Estado resultante.
        """
        return self.actions.get(state.name, {}).get(action.name)
    
    def action_cost(self, state: State, action: Action) -> float:
        """
        Devuelve el costo de aplicar una acción a un estado dado.
        
        Args:
            state (State): Estado inicial.
            action (Action): Acción a aplicar.
        
        Returns:
            float: Costo de la acción.
        """
        return self.costs.get(state.name, {}).get(action.name, float("inf"))