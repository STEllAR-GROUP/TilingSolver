import detail
import unittest

from util_test import make_basic_edge_set
from problem import Problem, Edge


class TestProblem(unittest.TestCase):
    def setUp(self):
        edge_set, vertex_sizes = make_basic_edge_set()
        self.problem = Problem(edge_set, vertex_sizes, 1)

    def test_creation(self):
        pass

    def test_level_sets(self):
        level_sets = detail.get_level_sets(self.problem.partial_order)
        self.assertEqual(level_sets, [set(['_begin_']), set(['add0', 'mul4', 'mul2']), set(['add3', 'add1'])])

    def test_get_tiling_tuples(self):
        tiling_matches = self.problem.get_tiling_tuples(2)
        self.assertEqual(set(tiling_matches), {('row', 'block'), ('block', 'col'),
                                               ('row', 'col'), ('block', 'block'),
                                               ('block', 'row'), ('col', 'block'),
                                               ('col', 'col'), ('col', 'row'),
                                               ('row', 'row')})

    def test_cost(self):
        cost = self.problem()
        print("Cost object output: ")
        print(cost)
        print("Cost object output finished")


if __name__ == '__main__':
    unittest.main()
