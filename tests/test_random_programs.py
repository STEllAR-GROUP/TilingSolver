import sys
import unittest

from detail import get_level_sets
from util_test import generate_random_problem, generate_entire_program, run_four_tests


class TestRandomPrograms(unittest.TestCase):
    MY_SEED = None

    def test_cost_exterior(self):
        self.MY_SEED = 101
        problem, inputs = generate_random_problem(self.MY_SEED)
        print(problem.edges)
        print(get_level_sets(problem.partial_order))
        print(generate_entire_program(inputs, problem))
        print(list(problem.partial_order.edges))

    def test_cost_small_vars_many_expressions(self):
        self.MY_SEED = 101
        problem, inputs = generate_random_problem(self.MY_SEED, 20, 4)
        print(problem.edges)
        print(get_level_sets(problem.partial_order))
        print(generate_entire_program(inputs, problem))
        print(list(problem.partial_order.edges))

    def test_problem_supplied_size_four_tests(self):
        self.MY_SEED = 101
        num_expressions = 80
        num_input_vars = 30
        problem, inputs = generate_random_problem(self.MY_SEED, num_expressions, num_input_vars)

        print(problem.edges)
        print(get_level_sets(problem.partial_order))
        print(generate_entire_program(inputs, problem))
        print(list(problem.partial_order.edges))
        run_four_tests([], [], verbosity=1, prob=problem, skip_real_exhaustive=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestRandomPrograms.MY_SEED = sys.argv.pop()
    TestRandomPrograms.MY_SEED = 100
    unittest.main()
