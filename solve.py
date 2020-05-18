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
        greedy_solver(prob, sub_hypergraph, sub_graph, tau)


def greedy_solver(problem, sub_hyper, sub_graph, tau):
    level_decomp = detail.get_level_sets(sub_graph)
    vars = [n for n, d in sub_hyper.nodes(data=True) if d['bipartite'] == 1]

    implementation_space_size = 1
    for edge in sub_graph.nodes():
        implementation_space_size *= problem.edges[edge].num_implementations()

    tiling_space_size = 3**len(vars)

    if implementation_space_size*tiling_space_size <= tau:
        print("Exhaustive search")
        return exhaust(problem, sub_hyper, sub_graph)
    i = 2
    print(i)

    if implementation_space_size <= tau:
        # This code is all wrong, I need a representation
        # for a solution
        min_solution = 0xefffffffffffffff
        for j in range(implementation_space_size):
            tmp = greedy_solve_helper(sub_hyper, sub_graph, level_decomp, implementation_space_size)
            if min_solution > tmp:
                min_solution = tmp
        return min_solution
    else:
        implementation_choices = {edge.edge_name: edge.random_imp() for edge in problem.edges.values()}
        for edge in sub_graph.nodes():
            # Choose implementation with minimal cost
            # TODO - This is a placeholder, put real implementation
            implementation_choices[edge.edge_name] = edge.random_imp()
        return implementation_choices, greedy_solve_helper(sub_hyper, sub_graph, level_decomp, implementation_choices)


def exhaust(problem, sub_hyper, sub_graph):
    vars_solution = {var_.var_name: var_ for var_ in
                     [problem.vertices[n] for n, d in sub_hyper.nodes(data=True) if d['bipartite'] == 1]}



def greedy_solve_helper(sub_hyper, sub_graph, level_decomp, implementation_choices):
    i = 2
    print(i)
    return i



