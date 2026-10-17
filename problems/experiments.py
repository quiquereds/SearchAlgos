from dataclasses import dataclass

from components.connectivity import assert_fully_connected
from problems.cable_installation import CableInstallationProblem
from problems.plans import PLAN_A, PLAN_B, ConduitPlan


@dataclass(frozen=True)
class Experiment:
    id: str
    plan: ConduitPlan
    demand: tuple[str, ...]
    human_reference_m: float

    def build(self) -> CableInstallationProblem:
        return CableInstallationProblem(self.plan, self.demand)

    def validate(self) -> None:
        """Falla si el plano tiene nodos desconectados o el tablero no alcanza alguna demanda."""
        assert_fully_connected(self.build().create_multi_objective_problem())


def _nodes(*ids: int) -> tuple[str, ...]:
    return tuple(str(i) for i in ids)


EXPERIMENTS = (
    Experiment("1.1", PLAN_A, _nodes(0, 1, 7, 15, 22, 23, 24), 30.4),
    Experiment(
        "1.2",
        PLAN_A,
        _nodes(0, 1, 2, 3, 6, 4, 5, 16, 17, 18, 19, 12, 13, 24, 29, 27, 26, 37, 36, 35, 33),
        120.57,
    ),
    Experiment("1.3", PLAN_A, _nodes(*range(38)), 136.85),
    Experiment("2.1", PLAN_B, _nodes(0, 1, 6, 2, 15, 24, 3, 4, 25, 14, 26, 27, 28, 5), 110.2),
    Experiment("2.2", PLAN_B, _nodes(*range(29)), 129.8),
    Experiment("2.3", PLAN_B, _nodes(0, 29, 33, 36, 39, 40, 41, 42, 52, 53, 54, 55, 56, 61), 130.0),
    # La lista original decía "45, 56, 47"; se asume 46 porque 0 y 29..61 suman los 34 nodos del artículo.
    Experiment("2.4", PLAN_B, _nodes(0, *range(29, 62)), 171.5),
)


def get_experiment(experiment_id: str) -> Experiment:
    for experiment in EXPERIMENTS:
        if experiment.id == experiment_id:
            return experiment
    known = ", ".join(e.id for e in EXPERIMENTS)
    raise KeyError(f"experimento desconocido: {experiment_id} (disponibles: {known})")
