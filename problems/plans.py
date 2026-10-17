"""Redes de conductos de los dos planos residenciales del artículo.

Cada plano es un grafo no dirigido y ponderado (pesos en metros de conducto).
Las aristas se listan una sola vez; el grafo completo se arma al construir el problema.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConduitPlan:
    name: str
    node_count: int
    panel: str
    edges: tuple[tuple[str, str, float], ...]

    def __post_init__(self) -> None:
        nodes = {str(i) for i in range(self.node_count)}
        seen: set[frozenset[str]] = set()
        for a, b, weight in self.edges:
            if a not in nodes or b not in nodes:
                raise ValueError(f"{self.name}: arista ({a}, {b}) fuera de los nodos 0..{self.node_count - 1}")
            if a == b:
                raise ValueError(f"{self.name}: la arista ({a}, {b}) es un lazo")
            if weight <= 0:
                raise ValueError(f"{self.name}: la arista ({a}, {b}) tiene peso no positivo ({weight})")
            key = frozenset((a, b))
            if key in seen:
                raise ValueError(f"{self.name}: arista ({a}, {b}) repetida")
            seen.add(key)
        if self.panel not in nodes:
            raise ValueError(f"{self.name}: el tablero {self.panel} no es un nodo del plano")

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(str(i) for i in range(self.node_count))


_PLAN_A_EDGES = (
    ("0", "1", 2.0),
    ("0", "20", 3.8),
    ("1", "2", 2.8),
    ("1", "7", 3.8),
    ("2", "8", 4.0),
    ("3", "9", 3.65),
    ("4", "10", 3.4),
    ("5", "11", 3.55),
    ("6", "8", 2.5),
    ("7", "15", 3.7),
    ("8", "9", 2.1),
    ("8", "15", 1.5),
    ("9", "10", 2.3),
    ("9", "16", 3.5),
    ("9", "22", 2.2),
    ("10", "11", 2.4),
    ("10", "17", 3.65),
    ("11", "14", 3.2),
    ("11", "18", 3.6),
    ("11", "23", 4.3),
    ("12", "23", 2.5),
    ("13", "23", 2.8),
    ("14", "19", 3.45),
    ("15", "22", 2.8),
    ("19", "25", 4.2),
    ("20", "21", 2.85),
    ("21", "22", 3.6),
    ("21", "33", 5.2),
    ("22", "27", 3.85),
    ("22", "30", 4.5),
    ("22", "32", 6.0),
    ("23", "24", 3.1),
    ("23", "28", 3.85),
    ("25", "29", 2.0),
    ("26", "30", 3.6),
    ("28", "32", 3.6),
    ("29", "32", 3.0),
    ("30", "31", 3.0),
    ("30", "36", 3.5),
    ("31", "32", 3.2),
    ("31", "34", 2.5),
    ("32", "34", 3.5),
    ("32", "37", 4.1),
    ("33", "35", 2.0),
    ("35", "36", 3.0),
)

_PLAN_B_EDGES = (
    ("0", "6", 4.8),
    ("0", "17", 6.2),
    ("0", "7", 3.8),
    ("0", "2", 5.2),
    ("1", "6", 3.5),
    ("2", "7", 4.7),
    ("3", "8", 4.8),
    ("4", "9", 5.2),
    ("5", "13", 4.2),
    ("6", "16", 4.5),
    ("7", "17", 3.2),
    ("7", "8", 3.4),
    ("8", "18", 3.2),
    ("8", "9", 4.1),
    ("9", "19", 3.2),
    ("9", "10", 2.5),
    ("10", "20", 3.2),
    ("10", "21", 5.8),
    ("10", "11", 3.8),
    ("11", "12", 3.5),
    ("12", "22", 3.5),
    ("12", "13", 3.1),
    ("13", "23", 3.5),
    ("14", "21", 3.8),
    ("15", "16", 4.3),
    ("16", "38", 4.5),
    ("16", "29", 5.5),
    ("17", "24", 3.3),
    ("17", "30", 3.8),
    ("17", "18", 3.5),
    ("18", "19", 4.2),
    ("19", "25", 3.7),
    ("19", "20", 2.5),
    ("21", "22", 3.7),
    ("21", "26", 3.7),
    ("22", "27", 3.5),
    ("22", "23", 3.2),
    ("23", "28", 4.2),
    ("23", "37", 3.6),
    ("29", "30", 3.4),
    ("30", "43", 5.9),
    ("30", "31", 4.5),
    ("31", "44", 5.3),
    ("31", "32", 3.9),
    ("32", "34", 3.8),
    ("32", "33", 4.2),
    ("34", "35", 3.5),
    ("35", "46", 5.8),
    ("36", "37", 3.7),
    ("37", "48", 5.6),
    ("38", "49", 5.2),
    ("39", "43", 3.6),
    ("40", "45", 3.5),
    ("41", "48", 4.2),
    ("42", "43", 3.9),
    ("43", "50", 3.5),
    ("43", "44", 4.3),
    ("44", "51", 3.6),
    ("44", "45", 4.5),
    ("45", "46", 3.5),
    ("45", "57", 4.2),
    ("46", "58", 4.3),
    ("46", "47", 3.8),
    ("47", "59", 3.7),
    ("47", "48", 3.2),
    ("48", "60", 3.7),
    ("49", "53", 4.8),
    ("49", "50", 5.2),
    ("50", "54", 4.2),
    ("50", "51", 4.4),
    ("52", "59", 3.8),
    ("54", "55", 3.9),
    ("56", "57", 3.8),
    ("57", "58", 3.5),
    ("58", "59", 4.2),
    ("59", "60", 3.2),
    ("60", "61", 2.5),
)


# En ambos planos el tablero de distribución es el nodo 0.
PLAN_A = ConduitPlan(name="A", node_count=38, panel="0", edges=_PLAN_A_EDGES)
PLAN_B = ConduitPlan(name="B", node_count=62, panel="0", edges=_PLAN_B_EDGES)
