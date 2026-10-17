from components.state import State


class Problem:
    """Problema de búsqueda sobre un grafo.

    actions[estado][acción] es el estado vecino al que lleva la acción y
    costs[estado][acción] su costo. Si no se dan costos, cada acción cuesta 1.
    """

    def __init__(
        self,
        initial: State,
        targets: list[State],
        actions: dict[str, dict[str, State]],
        costs: dict[str, dict[str, float]] | None = None,
    ):
        self.initial = initial
        self.targets = targets
        self.actions = actions
        self.costs = costs or {s: {a: 1.0 for a in acts} for s, acts in actions.items()}
