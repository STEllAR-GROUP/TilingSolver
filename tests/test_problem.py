import unittest
from problem import Problem


class TestProblem(unittest.TestCase):
    def setUp(self):
        edge_set = {'add_0': ('add', ['f', 'a', 'b'], 'row'),
                    'mul_0': ('mul', ['g', 'b', 'c'], 'row'),
                    'add_1': ('add', ['h', 'e', 'g'], 'row'),
                    'mul_1': ('mul', ['i', 'd', 'c'], 'row')}
        # [l_l, l_s, s_l, s_s]
        vertex_sizes = [['d'], ['c'], ['a', 'b'], ['e']]
        self.problem = Problem(edge_set, vertex_sizes, 1)

    def test_creation(self):
        pass

    def test_level_sets(self):
        level_sets = self.problem.get_level_sets()
        self.assertEqual(level_sets, [['_begin_'], ['add_0', 'mul_0', 'mul_1'], ['add_1']])

    def test_get_tiling_tuples(self):
        tiling_matches = self.problem.get_tiling_tuples(2)
        self.assertEqual(set(tiling_matches), {('row', 'block'), ('block', 'col'),
                                               ('row', 'col'), ('block', 'block'),
                                               ('block', 'row'), ('col', 'block'),
                                               ('col', 'col'), ('col', 'row'),
                                               ('row', 'row')})

    def test_cost(self):
        cost = self.problem()
        print(cost)


if __name__ == '__main__':
    unittest.main()
