import detail
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from edge import Edge
from problem import Problem


def get_sub_problem(prob, sub_hypergraph, sub_graph):
    extra = list(nx.isolates(sub_hypergraph))
    sub_hypergraph.remove_nodes_from(extra)
    vars = [n for n, d in sub_hypergraph.nodes(data=True) if d['bipartite'] == 1]
    edges = {edge.name: edge for edge in prob.edges.values() if edge.name in sub_graph.nodes}
    vert = {var.name: var for var in prob.vertices.values() if var.name in vars}

    sub_problem = Problem([], [], 1, edges=edges, vertices=vert,
                          hypergraph=sub_hypergraph, partial_order=sub_graph)
    return sub_problem


def solve(prob: Problem, tau=10, b=2, eta=0.1):
    components = nx.weakly_connected_components(prob.partial_order)
    comp = list(components)
    comp = [list(component) for component in comp]
    results = {component[0]: -1 for component in comp}
    for component in comp:
        print(component)
        sub_graph = prob.partial_order.subgraph(component)
        sub_hypergraph = prob.hypergraph.subgraph(prob.ground_set | set(component)).copy()
        results[component[0]] = greedy_solver(get_sub_problem(prob, sub_hypergraph, sub_graph), tau, b, eta)
    return results


def greedy_solver(problem, tau, b, eta):
    level_decomp = detail.get_level_sets(problem.partial_order)
    vars = [n for n, d in problem.hypergraph.nodes(data=True) if d['bipartite'] == 1]
    # Need to create new Problem from our sub_hyper and sub_graph
    # It's the only proper way to do this
    implementation_space_size = 1
    for edge in problem.edges:
        implementation_space_size *= problem.edges[edge].num_implementations()

    tiling_space_size = 3**len(vars)

    if implementation_space_size*tiling_space_size <= tau:
        print("Exhaustive search")
        vars_solution = [n for n, d in problem.hypergraph.nodes(data=True)
                         if d['bipartite'] == 1]
        edges_solution = [edge_name for edge_name in problem.edges]
        return exhaust(problem, vars_solution, edges_solution)
    else:
        print("S too big for exhaustive search at: ", implementation_space_size*tiling_space_size, " = ", implementation_space_size, "*", tiling_space_size)
        print("Number of vars: ", len(vars))

    if implementation_space_size <= tau:
        vars_solution = [problem.vertices[n] for n, d in problem.hypergraph.nodes(data=True)
                         if d['bipartite'] == 1]
        algorithm_choices = [problem.edges[edge_name] for edge_name in problem.edges]
        total_solution = vars_solution + algorithm_choices
        solution_map = {point.name: point.get_option() for point in algorithm_choices}
        finished = False
        best_solution = solution_map.copy()
        best_cost = problem.calculate_cost()
        count = 1
        # TODO - Move this into generic exhaustive search
        while not finished:
            finished = algorithm_choices[0].next(algorithm_choices)
            problem = greedy_solve_helper(problem, b, eta)
            tmp_cost = problem.calculate_cost()
            if tmp_cost < best_cost:
                best_cost = tmp_cost
                best_solution = {point.name: point.get_option() for point in total_solution}
                print("Reassignment: ", count, best_cost, best_solution)
            count += 1
        return best_cost, best_solution
    else:
        print("Minimum cost deviation method")
        for edge in problem.edges.values():
            edge.set_min_cost_deviance_algorithm()
        implementation_choices = {edge.name: edge.options[edge.idx] for edge in problem.edges.values()}
        problem = greedy_solve_helper(problem, b, eta)
        vars_solution = [problem.vertices[n] for n, d in problem.hypergraph.nodes(data=True)
                         if d['bipartite'] == 1]
        algorithm_choices = [problem.edges[edge_name] for edge_name in problem.edges]
        total_solution = vars_solution + algorithm_choices
        solution_map = {point.name: point.get_option() for point in total_solution}
        return problem.calculate_cost(), solution_map


def exhaust(problem, var_names, edge_names):
    print(var_names, edge_names)
    vars_solution = [problem.vertices[var_name] for var_name in var_names]
    edges_solution = [problem.edges[edge_name] for edge_name in edge_names]
    total_solution = vars_solution+edges_solution
    solution_map = {point.name: point.get_option() for point in total_solution}
    finished = len(total_solution) == 0
    best_solution = solution_map.copy()
    best_cost = problem.calculate_cost()
    count = 1
    while not finished:
        finished = total_solution[0].next(total_solution)
        tmp_cost = problem.calculate_cost()
        if tmp_cost < best_cost:
            best_cost = tmp_cost
            best_solution = {point.name: point.get_option() for point in total_solution}
            print("Reassignment: ", count, best_cost, best_solution)
        count += 1
    print("Total iterations: ", count)
    return best_cost, best_solution


def greedy_solve_helper(problem, b, eta):
    edges_prime = set(problem.edges.keys())
    decided_tiling = set()
    num_vars = len(problem.vertices)
    while len(decided_tiling) < num_vars:
        edge_bucket = compute_greedy_order(problem, edges_prime, decided_tiling, b, eta)
        t_prime = set()
        for edge_name in edge_bucket:
            for var_name in problem.edges[edge_name].vars:
                if var_name not in decided_tiling:
                    t_prime.add(var_name)
        exhaust(problem, t_prime, [])
        decided_tiling = decided_tiling | t_prime
        edges_prime -= edge_bucket
    return problem


def get_edge_min_cost(problem, edge_name, decided_tiling):
    edge = problem.edges[edge_name]
    finished = False
    tmp_min = 99999999
    var_set = [problem.vertices[var_name] for var_name in edge.vars if var_name not in decided_tiling]

    if len(var_set) > 0:
        while not finished:
            finished = var_set[0].next(var_set)
            tmp_cost = problem.calculate_edge_subset_cost([edge.name])
            tmp_min = min(tmp_min, tmp_cost)
    return tmp_min


def sum_descendants(problem, edge_name, gamma):
    edge_sum = 0
    for edge_name in problem.partial_order.successors(edge_name):
        if edge_name in gamma:
            edge_sum += gamma[edge_name]
    return edge_sum


def compute_greedy_order(problem, edges_prime, decided_tiling, b, eta):
    gamma = {}
    level_sets = detail.get_level_sets(problem.partial_order)
    last_level_set = level_sets[-1]
    for edge_name in last_level_set & edges_prime:
        gamma[edge_name] = get_edge_min_cost(problem, edge_name, decided_tiling)

    for level_set in reversed(level_sets[1:-1]):
        for edge_name in level_set & edges_prime:
            gamma[edge_name] = get_edge_min_cost(problem, edge_name, decided_tiling) \
                               + sum_descendants(problem, edge_name, gamma)
    assert len(gamma) == len(edges_prime), "Something went wrong with compute greedy order"
    i = 0
    sorted_eps = sorted(list(gamma.items()))
    edges_double_prime = set()
    while len(edges_double_prime) <= b and i < len(sorted_eps):
        if sorted_eps[i][1] >= eta*sorted_eps[0][1]:
            edges_double_prime.add(sorted_eps[i][0])
        else:
            return edges_double_prime
        i += 1
    return edges_double_prime


