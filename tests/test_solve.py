import unittest

from problem import Problem
from solve import solve
from util_test import make_basic_edge_set


class TestSolver(unittest.TestCase):
    def setUp(self):
        edge_set, vertex_sizes = make_basic_edge_set()
        self.problem = Problem(edge_set, vertex_sizes, 1)

    def test_solve(self):
        # tau is arbitrary right now
        solve(self.problem, tau=10)


if __name__ == '__main__':
    unittest.main()
