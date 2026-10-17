import random
import unittest

from algorithms.steiner_exact import steiner_tree_cost
from algorithms.steiner_solution import connects
from algorithms.takahashi_matsuyama import takahashi_matsuyama
from problems.experiments import get_experiment
from tests.test_dijkstra_multi_goal import random_connected_graph

try:
    from algorithms.steiner_ilp import steiner_ilp
except ImportError:
    steiner_ilp = None  # type: ignore[assignment]

TOLERANCE = 1e-9
COUNTEREXAMPLE = [("s", "h", 6.0), ("h", "d1", 4.0), ("h", "d2", 4.0), ("s", "d1", 9.0), ("d1", "d2", 7.0)]


def nodes_of(edges):
    return sorted({n for a, b, _ in edges for n in (a, b)})


class TakahashiMatsuyama(unittest.TestCase):
    def test_counterexample_cost_and_validity(self):
        solution = takahashi_matsuyama(nodes_of(COUNTEREXAMPLE), COUNTEREXAMPLE, "s", ["s", "d1", "d2"])
        self.assertTrue(connects(solution, ["s", "d1", "d2"]))
        self.assertAlmostEqual(solution.cost, 16.0)

    def test_unreachable_terminal_returns_none(self):
        edges = [("a", "b", 1.0), ("c", "d", 1.0)]
        self.assertIsNone(takahashi_matsuyama(nodes_of(edges), edges, "a", ["a", "d"]))

    def test_root_only_costs_nothing(self):
        solution = takahashi_matsuyama(["a", "b"], [("a", "b", 2.0)], "a", ["a"])
        self.assertEqual((solution.edges, solution.cost), ((), 0))

    def test_within_twice_the_optimum_on_random_graphs(self):
        rng = random.Random(11)
        for _ in range(120):
            size = rng.randint(4, 10)
            edges = random_connected_graph(rng, size)
            terminals = [str(i) for i in rng.sample(range(size), rng.randint(2, min(6, size)))]
            solution = takahashi_matsuyama(nodes_of(edges), edges, terminals[0], terminals)
            optimum = steiner_tree_cost(nodes_of(edges), edges, terminals)
            self.assertTrue(connects(solution, terminals))
            self.assertGreaterEqual(solution.cost, optimum - TOLERANCE)
            self.assertLessEqual(solution.cost, 2 * optimum + TOLERANCE)


@unittest.skipIf(steiner_ilp is None, "scipy no está instalado")
class ExactIlp(unittest.TestCase):
    def test_counterexample_optimum(self):
        result = steiner_ilp(nodes_of(COUNTEREXAMPLE), COUNTEREXAMPLE, "s", ["s", "d1", "d2"])
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.solution.cost, 14.0)

    def test_matches_dreyfus_wagner_on_random_graphs(self):
        rng = random.Random(5)
        for _ in range(60):
            size = rng.randint(4, 10)
            edges = random_connected_graph(rng, size)
            terminals = [str(i) for i in rng.sample(range(size), rng.randint(2, min(6, size)))]
            result = steiner_ilp(nodes_of(edges), edges, terminals[0], terminals)
            self.assertEqual(result.status, "optimal")
            self.assertTrue(connects(result.solution, terminals))
            self.assertAlmostEqual(result.solution.cost, steiner_tree_cost(nodes_of(edges), edges, terminals))

    def test_matches_dreyfus_wagner_on_small_experiments(self):
        for experiment_id in ("1.1", "2.1", "2.3"):
            with self.subTest(experiment=experiment_id):
                experiment = get_experiment(experiment_id)
                plan = experiment.plan
                terminals = list(dict.fromkeys((plan.panel, *experiment.demand)))
                result = steiner_ilp(plan.nodes, plan.edges, plan.panel, terminals)
                self.assertEqual(result.status, "optimal")
                self.assertAlmostEqual(result.solution.cost, steiner_tree_cost(plan.nodes, plan.edges, terminals))


if __name__ == "__main__":
    unittest.main()
