import detail
import networkx as nx
import unittest

from util_test import make_basic_edge_set
from problem import Problem


class TestProblem(unittest.TestCase):
    def setUp(self):
        edge_set, vertex_sizes = make_basic_edge_set()
        self.problem = Problem(edge_set, vertex_sizes, 1)

    def get_sub_problem(self, edges_subset):
        sub_graph = self.problem.partial_order.subgraph(edges_subset)
        sub_hypergraph = self.problem.hypergraph.subgraph(self.problem.ground_set | set(edges_subset)).copy()
        extra = list(nx.isolates(sub_hypergraph))
        sub_hypergraph.remove_nodes_from(extra)
        vars = [n for n, d in sub_hypergraph.nodes(data=True) if d['bipartite'] == 1]
        edges = {edge.name: edge for edge in self.problem.edges.values() if edge.name in edges_subset}
        vert = {var.name: var for var in self.problem.vertices.values() if var.name in vars}
        second_problem = Problem([], [], 1, edges=edges, vertices=vert,
                                 hypergraph=sub_hypergraph, partial_order=sub_graph)
        return second_problem

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

    def test_sub_problem(self):
        sub_edges = ['_begin_', 'add0', 'mul2']
        self.get_sub_problem(sub_edges)

    def test_sub_problem_level_sets(self):
        sub_edges = ['_begin_', 'add0', 'mul2']
        sub_prob = self.get_sub_problem(sub_edges)
        level_sets = detail.get_level_sets(sub_prob.partial_order)
        self.assertEqual(level_sets, [set(['_begin_']), set(['add0', 'mul2'])])


    def test_cost(self):
        cost = self.problem()
        print("Cost object output: ")
        print(cost)
        print("Cost object output finished")


if __name__ == '__main__':
    unittest.main()
