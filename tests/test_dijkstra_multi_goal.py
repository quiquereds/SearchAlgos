import itertools
import random
import unittest

from algorithms.dijkstra_multi_goal import dijkstra_multi_goal
from algorithms.steiner_exact import steiner_tree_cost
from components.problem import Problem
from components.state import State

TOLERANCE = 1e-9


def make_problem(edges, start, demand) -> Problem:
    names = sorted({n for a, b, _ in edges for n in (a, b)} | {start} | set(demand))
    states = {name: State(name) for name in names}
    actions: dict[str, dict[str, State]] = {name: {} for name in names}
    costs: dict[str, dict[str, float]] = {name: {} for name in names}
    for a, b, weight in edges:
        for source, destination in ((a, b), (b, a)):
            actions[source][f"to{destination}"] = states[destination]
            costs[source][f"to{destination}"] = weight
    return Problem(
        initial=states[start],
        targets=[states[d] for d in demand],
        actions=actions,
        costs=costs,
    )


def random_connected_graph(rng: random.Random, size: int):
    edges = {}
    for node in range(1, size):
        edges[(str(rng.randrange(node)), str(node))] = float(rng.randint(1, 9))
    for _ in range(size):
        a, b = rng.sample(range(size), 2)
        edges.setdefault((str(min(a, b)), str(max(a, b))), float(rng.randint(1, 9)))
    return [(a, b, w) for (a, b), w in edges.items()]


def brute_force_guiding_cost(edges, start, demand) -> float:
    """Mínimo g sobre todas las caminatas que visitan la demanda: TSP abierto sobre la clausura métrica."""
    names = sorted({n for a, b, _ in edges for n in (a, b)})
    dist = {(u, v): 0.0 if u == v else float("inf") for u in names for v in names}
    for a, b, weight in edges:
        dist[a, b] = dist[b, a] = min(dist[a, b], weight)
    for k in names:
        for u in names:
            for v in names:
                dist[u, v] = min(dist[u, v], dist[u, k] + dist[k, v])
    pending = [d for d in demand if d != start]
    best = float("inf")
    for order in itertools.permutations(pending):
        stops = (start, *order)
        best = min(best, sum(dist[a, b] for a, b in zip(stops, stops[1:])))
    return best


class ArticleCounterexample(unittest.TestCase):
    """Instancia de la sección 3.6: minimizar g no minimiza c."""

    EDGES = [("s", "h", 6), ("h", "d1", 4), ("h", "d2", 4), ("s", "d1", 9), ("d1", "d2", 7)]

    def test_search_returns_walk_through_d1_and_d2(self):
        result = dijkstra_multi_goal(make_problem(self.EDGES, "s", ["d1", "d2"]))
        self.assertEqual(result.walk, ("s", "d1", "d2"))
        self.assertAlmostEqual(result.guiding_cost, 16)
        self.assertAlmostEqual(result.cost, 16)

    def test_exact_optimum_is_lower(self):
        nodes = ["s", "h", "d1", "d2"]
        self.assertAlmostEqual(steiner_tree_cost(nodes, self.EDGES, ["s", "d1", "d2"]), 14)


class SearchContract(unittest.TestCase):
    def test_start_as_only_demand_costs_nothing(self):
        result = dijkstra_multi_goal(make_problem([("a", "b", 2.0)], "a", ["a"]))
        self.assertEqual(result.walk, ("a",))
        self.assertEqual(result.cost, 0)
        self.assertEqual(result.states_expanded, 1)

    def test_single_demand_is_shortest_path(self):
        edges = [("a", "b", 1.0), ("b", "c", 1.0), ("a", "c", 5.0)]
        result = dijkstra_multi_goal(make_problem(edges, "a", ["c"]))
        self.assertEqual(result.walk, ("a", "b", "c"))
        self.assertAlmostEqual(result.cost, 2.0)

    def test_unreachable_demand_returns_none(self):
        problem = make_problem([("a", "b", 1.0), ("c", "d", 1.0)], "a", ["b", "d"])
        self.assertIsNone(dijkstra_multi_goal(problem))

    def test_shared_segment_is_counted_once(self):
        edges = [("s", "m", 3.0), ("m", "x", 2.0), ("m", "y", 2.0)]
        result = dijkstra_multi_goal(make_problem(edges, "s", ["x", "y"]))
        self.assertAlmostEqual(result.cost, 7.0)
        self.assertGreater(result.guiding_cost, result.cost)

    def test_progress_callback_reports_expansions(self):
        rng = random.Random(7)
        edges = random_connected_graph(rng, 40)
        calls = []
        dijkstra_multi_goal(
            make_problem(edges, "0", [str(i) for i in range(0, 40, 3)]),
            progress_callback=lambda expanded, g: calls.append(expanded),
        )
        self.assertTrue(calls)
        self.assertEqual(calls, sorted(calls))


class ArticlePropositions(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(2026)

    def random_instance(self):
        size = self.rng.randint(4, 9)
        edges = random_connected_graph(self.rng, size)
        demand = [str(i) for i in self.rng.sample(range(size), self.rng.randint(1, min(5, size)))]
        start = str(self.rng.randrange(size))
        return edges, start, demand

    def test_proposition_1_search_is_optimal_for_guiding_cost(self):
        for _ in range(150):
            edges, start, demand = self.random_instance()
            result = dijkstra_multi_goal(make_problem(edges, start, demand))
            self.assertAlmostEqual(result.guiding_cost, brute_force_guiding_cost(edges, start, demand))

    def test_proposition_2_cost_chain(self):
        for _ in range(150):
            edges, start, demand = self.random_instance()
            result = dijkstra_multi_goal(make_problem(edges, start, demand))
            nodes = sorted({n for a, b, _ in edges for n in (a, b)})
            optimum = steiner_tree_cost(nodes, edges, [start, *demand])
            self.assertLessEqual(optimum, result.cost + TOLERANCE)
            self.assertLessEqual(result.cost, result.guiding_cost + TOLERANCE)
            self.assertLessEqual(result.guiding_cost, 2 * optimum + TOLERANCE)

    def test_reported_edges_cover_start_and_demand(self):
        for _ in range(100):
            edges, start, demand = self.random_instance()
            result = dijkstra_multi_goal(make_problem(edges, start, demand))
            touched = {n for edge in result.edges for n in edge} | {start}
            self.assertTrue(set(demand) <= touched)


if __name__ == "__main__":
    unittest.main()
