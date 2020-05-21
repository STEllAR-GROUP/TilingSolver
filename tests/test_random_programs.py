import edge
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import random
import sys
import unittest

from detail import get_edge_types, get_level_sets, get_valid_input_lists, get_output_size_calculators
from edge import Edge
from matrix_size import MatrixSize
from problem import Problem
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
        #print((nx.connected_components(problem.partial_order)))



if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestRandomPrograms.MY_SEED = sys.argv.pop()
    TestRandomPrograms.MY_SEED = 100
    unittest.main()
