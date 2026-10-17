import unittest

from components.route_format import format_tree, format_walk

WEIGHTS = {
    frozenset(("0", "1")): 2.0,
    frozenset(("0", "2")): 3.0,
    frozenset(("1", "3")): 4.0,
    frozenset(("2", "3")): 1.5,
}


class FormatTree(unittest.TestCase):
    def test_draws_branches_with_lengths_and_marks_demand(self):
        edges = [("0", "1"), ("0", "2"), ("1", "3")]
        expected = "\n".join(["0", "├─ 1 (2 m)", "│  └─ 3* (4 m)", "└─ 2 (3 m)"])
        self.assertEqual(format_tree(edges, "0", WEIGHTS, demand={"3"}), expected)

    def test_children_are_ordered_numerically(self):
        weights = {frozenset(("0", "2")): 1.0, frozenset(("0", "10")): 1.0}
        lines = format_tree([("0", "10"), ("0", "2")], "0", weights).splitlines()
        self.assertEqual([line.split()[1] for line in lines[1:]], ["2", "10"])

    def test_cycle_edges_are_reported(self):
        edges = [("0", "1"), ("0", "2"), ("1", "3"), ("2", "3")]
        text = format_tree(edges, "0", WEIGHTS)
        self.assertIn("Aristas que cierran ciclos: 0–2", text)

    def test_root_only(self):
        self.assertEqual(format_tree([], "0", {}, demand={"0"}), "0*")


class FormatWalk(unittest.TestCase):
    def test_joins_nodes_with_arrows(self):
        self.assertEqual(format_walk(["0", "1", "7"]), "0 → 1 → 7")

    def test_wraps_long_walks(self):
        text = format_walk([str(i) for i in range(80)], width=40)
        self.assertGreater(len(text.splitlines()), 1)
        self.assertTrue(all(len(line) <= 40 for line in text.splitlines()))


if __name__ == "__main__":
    unittest.main()
