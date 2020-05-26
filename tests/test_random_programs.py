import sys
import unittest

from detail import get_level_sets
from util_test import generate_random_problem, generate_entire_program


class TestRandomPrograms(unittest.TestCase):
    MY_SEED = None

    def test_cost_exterior(self):
        self.MY_SEED = 101
        problem, inputs = generate_random_problem(self.MY_SEED)
        print(problem.edges)
        print(get_level_sets(problem.partial_order))
        print(generate_entire_program(inputs, problem))
        print(type(problem.partial_order))
        #nx.draw(problem.partial_order)
        #plt.show()
        print(list(problem.partial_order.edges))

    def test_cost_small_vars_many_expressions(self):
        self.MY_SEED = 101
        problem, inputs = generate_random_problem(self.MY_SEED, num_expressions=20, num_input_vars=4)
        print(problem.edges)
        print(get_level_sets(problem.partial_order))
        print(generate_entire_program(inputs, problem))
        print(type(problem.partial_order))
        #nx.draw(problem.partial_order)
        #plt.show()
        print(list(problem.partial_order.edges))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestRandomPrograms.MY_SEED = sys.argv.pop()
    TestRandomPrograms.MY_SEED = 100
    unittest.main()
