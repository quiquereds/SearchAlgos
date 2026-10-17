import unittest

from algorithms.dijkstra_multi_goal import dijkstra_multi_goal
from components.connectivity import DisconnectedGraphError
from problems.cable_installation import CableInstallationProblem
from problems.experiments import EXPERIMENTS, Experiment, get_experiment
from problems.plans import PLAN_A, PLAN_B, ConduitPlan

DEMAND_SIZES = {"1.1": 7, "1.2": 21, "1.3": 38, "2.1": 14, "2.2": 29, "2.3": 14, "2.4": 34}


class Plans(unittest.TestCase):
    def test_plan_sizes(self):
        self.assertEqual((PLAN_A.node_count, len(PLAN_A.edges)), (38, 45))
        self.assertEqual((PLAN_B.node_count, len(PLAN_B.edges)), (62, 77))

    def test_plan_rejects_repeated_edge(self):
        with self.assertRaises(ValueError):
            ConduitPlan("X", 2, "0", (("0", "1", 1.0), ("1", "0", 1.0)))

    def test_plan_rejects_unknown_node(self):
        with self.assertRaises(ValueError):
            ConduitPlan("X", 2, "0", (("0", "5", 1.0),))


class Experiments(unittest.TestCase):
    def test_demand_sizes_match_the_definitions(self):
        self.assertEqual({e.id: len(e.demand) for e in EXPERIMENTS}, DEMAND_SIZES)

    def test_every_experiment_is_fully_connected(self):
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment.id):
                experiment.validate()

    def test_panel_is_a_demand_node_in_every_experiment(self):
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment.id):
                self.assertIn(experiment.plan.panel, experiment.demand)

    def test_repeated_demand_is_rejected(self):
        with self.assertRaises(ValueError):
            CableInstallationProblem(PLAN_A, ["0", "1", "1"])

    def test_isolated_node_is_detected(self):
        plan = ConduitPlan("X", 3, "0", (("0", "1", 1.0),))
        with self.assertRaises(DisconnectedGraphError):
            Experiment("x", plan, ("0", "1"), 1.0).validate()

    def test_unknown_experiment(self):
        with self.assertRaises(KeyError):
            get_experiment("9.9")

    def test_experiment_1_1_regression(self):
        problem = get_experiment("1.1").build().create_multi_objective_problem()
        result = dijkstra_multi_goal(problem)
        self.assertAlmostEqual(result.cost, 26.60)
        self.assertTrue(result.is_tree)


if __name__ == "__main__":
    unittest.main()
