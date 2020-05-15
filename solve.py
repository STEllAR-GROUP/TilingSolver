import detail
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from edge import Edge
from problem import Problem

def solve(prob: Problem, tau=10):
    i = 2
    i += 1
    components = nx.weakly_connected_components(prob.partial_order)
    comp = list(components)
    for component in comp:
        sub_graph = prob.partial_order.subgraph(component)
        sub_hypergraph = prob.hypergraph.subgraph(prob.ground_set | component)
        #nx.draw(sub_graph)
        #plt.show()
        greedy_solver(sub_hypergraph, sub_graph, tau)


def greedy_solver(sub_hyper, sub_graph, tau):
    level_decomp = detail.get_level_sets(sub_graph)

    i = 2
    print(i)



def greedy_solve_helper(sub_hyper, sub_graph, level_decomp, impl):
    i = 2
    print(i)



