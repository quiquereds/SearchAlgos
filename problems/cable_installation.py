from collections.abc import Sequence

from components.problem import Problem
from components.state import State
from problems.plans import ConduitPlan


class CableInstallationProblem:
    """Un plano de conductos junto con el conjunto de nodos demanda a energizar."""

    def __init__(self, plan: ConduitPlan, demand_nodes: Sequence[str]):
        unknown = [n for n in demand_nodes if n not in plan.nodes]
        if unknown:
            raise ValueError(f"nodos demanda fuera del plano {plan.name}: {unknown}")
        if len(set(demand_nodes)) != len(demand_nodes):
            repeated = sorted({n for n in demand_nodes if list(demand_nodes).count(n) > 1})
            raise ValueError(f"nodos demanda repetidos: {repeated}")

        self.plan = plan
        self.target_nodes = list(demand_nodes)
        self.network_nodes = {name: State(name) for name in plan.nodes}
        self.actions, self.costs = self._build_network()

    def _build_network(self):
        actions = {name: {} for name in self.network_nodes}
        costs = {name: {} for name in self.network_nodes}
        for a, b, length in self.plan.edges:
            for source, destination in ((a, b), (b, a)):
                actions[source][f"to{destination}"] = self.network_nodes[destination]
                costs[source][f"to{destination}"] = length
        return actions, costs

    def create_multi_objective_problem(self) -> Problem:
        return Problem(
            initial=self.network_nodes[self.plan.panel],
            targets=[self.network_nodes[name] for name in self.target_nodes],
            actions=self.actions,
            costs=self.costs,
        )

    def create_single_objective_problem(self, target_node: str) -> Problem:
        """Mismo plano con un único destino, para algoritmos de origen-destino."""
        return Problem(
            initial=self.network_nodes[self.plan.panel],
            targets=[self.network_nodes[target_node]],
            actions=self.actions,
            costs=self.costs,
        )

    def get_problem_summary(self) -> dict:
        lengths = [length for _, _, length in self.plan.edges]
        return {
            "total_nodes": len(self.network_nodes),
            "total_connections": len(self.plan.edges),
            "target_nodes": len(self.target_nodes),
            "target_list": self.target_nodes,
            "average_cable_cost": sum(lengths) / len(lengths),
        }
