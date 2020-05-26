import unittest

from problem import Problem
from solve import solve, trivial_solve
from util_test import make_basic_edge_set, generate_random_problem, \
    generate_entire_program, make_three_level_edge_set, \
    make_multi_component_edge_set


class TestSolver(unittest.TestCase):
    def setUp(self):
        edge_set, vertex_sizes = make_basic_edge_set()
        self.problem = Problem(edge_set, vertex_sizes, 1)

    def run_problem(self, problem, tau=10, tau_prime=20,
                    b=2, eta=0.1, trivial=False):
        if trivial:
            result = trivial_solve(problem)
        else:
            result = solve(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta)
        print("-----------------------------")
        print("Result: ", result)
        print("-----------------------------")

    def run_basic_problem(self, tau=10, tau_prime=20, b=2, eta=0.1,
                          trivial=False):
        problem = Problem(*make_basic_edge_set(), 1)
        self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    def run_three_level_problem(self, tau=10, tau_prime=20, b=2, eta=0.1,
                          trivial=False):
        problem = Problem(*make_three_level_edge_set(), 1)
        self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    def run_two_comp_problem(self, tau=10, tau_prime=20, b=2, eta=0.1, trivial=False):
        problem = Problem(*make_multi_component_edge_set(), 1)
        self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    def run_random_problem(self, seed=None, tau=10, tau_prime=20,
                           b=2, eta=0.1, num_expressions=None, num_input_vars=None, trivial=False):
        problem, inputs = generate_random_problem(seed, num_expressions=num_expressions, num_input_vars=num_input_vars)
        self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)
        print(generate_entire_program(inputs, problem))

    def test_trivial_solve(self):
        self.run_basic_problem(trivial=True)

    def test_solve_implementation_search(self):
        self.run_basic_problem()

    def test_solve_exhaustive(self):
        self.run_basic_problem(tau=90000)

    def test_solve_min_deviance(self):
        # tau is arbitrary right now
        self.run_basic_problem(tau_prime=2)

    def test_trivial_three_solve(self):
        self.run_three_level_problem(trivial=True)

    def test_three_solve_implementation_search(self):
        self.run_three_level_problem()

    def test_three_solve_exhaustive(self):
        self.run_three_level_problem(tau=90000)

    def test_three_solve_min_deviance(self):
        self.run_three_level_problem(tau_prime=2)

    def test_two_comp_trivial(self):
        self.run_two_comp_problem(trivial=True)

    def test_two_comp_impl_search(self):
        self.run_two_comp_problem(tau_prime=100)

    def test_two_comp_exhaustive(self):
        self.run_two_comp_problem(tau=600000)

    def test_two_com_min_deviance(self):
        self.run_two_comp_problem(tau_prime=2)

    def test_bigger_program_min_deviance(self):
        # 28.0, 0.079 sec
        self.run_random_problem(seed=101, tau=2, b=2, eta=0.1)

    def test_bigger_program_implementation_search(self):
        # 16.0, 26 sec
        self.run_random_problem(seed=101, tau_prime=80000, b=2, eta=0.1)

    def test_bigger_program_trivial(self):
        # 86.0, 0.005 sec
        self.run_random_problem(seed=101, trivial=True)

    def test_skinny_program_trivial(self):
        # Solution is 49.0, for 0.006 sec
        self.run_random_problem(seed=101, num_expressions=20,
                                num_input_vars=4, trivial=True)

    def test_skinny_program_implementation_search(self):
        # Solution is 38.0, six minutes
        self.run_random_problem(seed=101, num_expressions=20,
                                num_input_vars=4,
                                tau_prime=80000, b=2, eta=0.1)

    def test_skinny_program_min_deviance(self):
        # Solution is 68.0, 1/5th of second
        self.run_random_problem(seed=101, num_expressions=20,
                                num_input_vars=4,
                                tau_prime=2, b=2, eta=0.1)


if __name__ == '__main__':
    unittest.main()
