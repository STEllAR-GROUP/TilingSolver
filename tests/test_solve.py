import unittest

from problem import Problem
from solve import solve
from util_test import make_basic_edge_set, generate_random_problem, generate_entire_program



class TestSolver(unittest.TestCase):
    def setUp(self):
        edge_set, vertex_sizes = make_basic_edge_set()
        self.problem = Problem(edge_set, vertex_sizes, 1)

    def test_solve(self):
        # tau is arbitrary right now
        result = solve(self.problem, tau=10)
        print("-----------------------------")
        print("Result: ", result)
        print("-----------------------------")

    def test_solve_exhaustive(self):
        result = solve(self.problem, tau=90000)
        print("-----------------------------")
        print("Result: ", result)
        print("-----------------------------")


    def test_solve_min_deviance(self):
        # tau is arbitrary right now
        result = solve(self.problem, tau=2, b=2, eta=0.1)
        print("-----------------------------")
        print("Result: ", result)
        print("-----------------------------")

    def test_bigger_program(self):
        problem, inputs = generate_random_problem(101)
        result = solve(problem, tau=2, b=2, eta=0.1)
        print("-----------------------------")
        print("Result: ", result)
        print("-----------------------------")
        print(generate_entire_program(inputs, problem))



if __name__ == '__main__':
    unittest.main()
