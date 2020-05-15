import detail
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from problem import Problem, Edge

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
        greedy_solve(sub_hypergraph, sub_graph, tau)


def greedy_solve(sub_hyper, sub_graph, tau):
    level_decomp = detail.get_level_sets(sub_graph)
    i = 2
    print(i)
