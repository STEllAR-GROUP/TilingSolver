import copy
import detail
import networkx as nx
import numpy as np
import sys
import time

from problem import Problem

from math import log, floor
from multiprocessing import cpu_count
from joblib import Parallel, delayed


def get_sub_problem(prob, sub_hypergraph, sub_graph):
    extra = list(nx.isolates(sub_hypergraph))
    sub_hypergraph.remove_nodes_from(extra)
    vars = [n for n, d in sub_hypergraph.nodes(data=True) if d['bipartite'] == 1]
    edges = {edge.name: edge for edge in prob.edges.values() if edge.name in sub_graph.nodes}
    vert = {var.name: var for var in prob.variables.values() if var.name in vars}

    input_vars = [x for x in prob.input_vars if x in vars]

    sub_problem = Problem([], [], 1, edges=edges, variables=vert,
                          hypergraph=sub_hypergraph, partial_order=sub_graph, input_vars=input_vars)
    return sub_problem


def local_solve(prob: Problem):
    # TODO - Change name of trivial init
    prob = find_local_solution(prob)
    cost = prob.calculate_cost()
    vars_solution = list(prob.variables.values())
    algorithm_choices = list(prob.edges.values())
    total_solution = vars_solution + algorithm_choices
    solution_map = {point.name: point.get_option() for point in total_solution}
    return cost-len(prob.input_vars), solution_map


def greedy_solve(prob: Problem, tau=10, tau_prime=20, b=2, eta=0.1, verbosity=0, skip_real_exhaustive=False):
    graph = prob.partial_order.copy()
    graph.remove_node('_begin_')
    comps = nx.connected_components(prob.hypergraph)
    comp = [list(component) for component in comps]
    component_names = ["component_"+str(i) for i in range(len(comp))]
    results = {component_name: -1 for component_name in component_names}
    #print(graph.nodes, graph.edges)
    if verbosity > 0:
        print("Num. of Components: ", len(comp))
    for i in range(len(comp)):
        #print(i)
        component = comp[i]
        sub_graph = prob.partial_order.subgraph(list(set(component) | {'_begin_'})).copy()
        sub_hypergraph = prob.hypergraph.subgraph(prob.ground_set | set(component)).copy()
        isolates = list(nx.isolates(sub_hypergraph))
        sub_hypergraph.remove_nodes_from(isolates)
        results[component_names[i]] = greedy_solver(get_sub_problem(prob, sub_hypergraph, sub_graph), tau, tau_prime, b, eta, verbosity, skip_real_exhaustive=skip_real_exhaustive)
    return results


def greedy_solver(problem, tau, tau_prime, b, eta, verbosity, skip_real_exhaustive=False):
    vars = [n for n, d in problem.hypergraph.nodes(data=True) if d['bipartite'] == 1]
    # Need to create new Problem from our sub_hyper and sub_graph
    # It's the only proper way to do this
    implementation_space_size = 1
    for edge in problem.edges:
        implementation_space_size *= problem.edges[edge].num_implementations()

    tiling_space_size = 3**len(vars)

    if implementation_space_size*tiling_space_size <= tau:
        if verbosity > 0:
            print("Exhaustive search")
        vars_solution = [n for n, d in problem.hypergraph.nodes(data=True)
                         if d['bipartite'] == 1]
        edges_solution = [edge_name for edge_name in problem.edges]
        a, b = exhaust(problem, vars_solution, edges_solution, implementation_space_size*tiling_space_size, verbosity, skip_real_exhaustive=skip_real_exhaustive)
        return a-len(problem.input_vars), b
    elif verbosity > 0:
        print("S too big for exhaustive search at: ", implementation_space_size*tiling_space_size, " = ", implementation_space_size, "*", tiling_space_size)
        print("Number of vars: ", len(vars))

    if implementation_space_size <= tau_prime:
        if verbosity > 0:
            print("Exhaustive search over implementation space")
        vars_solution = [problem.variables[n] for n, d in problem.hypergraph.nodes(data=True)
                         if d['bipartite'] == 1]
        algorithm_choices = [problem.edges[edge_name] for edge_name in problem.edges]
        total_solution = vars_solution + algorithm_choices
        solution_map = {point.name: point.get_option() for point in total_solution}
        finished = False
        best_solution = solution_map.copy()
        best_cost = problem.calculate_cost()
        count = 1
        while not finished:
            finished = algorithm_choices[0].next(algorithm_choices)
            problem = greedy_solve_helper(problem, b, eta)
            tmp_cost = problem.calculate_cost()
            if tmp_cost < best_cost:
                best_cost = tmp_cost
                best_solution = {point.name: point.get_option() for point in total_solution}
                if verbosity > 0:
                    print("Reassignment upper level: ", count, best_cost, best_solution)
            count += 1
        if verbosity > 0:
            print("Best cost: ", best_cost)
        return best_cost-len(problem.input_vars), best_solution
    else:
        if verbosity > 0:
            print("Minimum cost deviation method")
        for edge in problem.edges.values():
            edge.set_min_cost_deviance_algorithm()
        problem = greedy_solve_helper(problem, b, eta)
        vars_solution = [problem.variables[n] for n, d in problem.hypergraph.nodes(data=True)
                         if d['bipartite'] == 1]
        algorithm_choices = [problem.edges[edge_name] for edge_name in problem.edges]
        total_solution = vars_solution + algorithm_choices
        solution_map = {point.name: point.get_option() for point in total_solution}
        return problem.calculate_cost()-len(problem.input_vars), solution_map


def find_local_solution(problem):
    assigned = {var_name: False for var_name in problem.variables}
    retile_cost = 0.0
    for level_set in detail.get_level_sets(problem.partial_order)[1:]:
        level_set_sortable = [(problem.edges[edge_name].program_index, edge_name) for edge_name in level_set]
        level_set_sortable.sort()
        for index, edge_name in level_set_sortable:
            edge = problem.edges[edge_name]
            cost_dict = edge.get_cost_dict()
            # Basically MAX_INT
            min_cost = sys.maxsize
            print(edge)
            local_vars = [problem.variables[var_name] for var_name in edge.vars]
            for alg in edge.options:
                cost_table = cost_dict[alg]()
                #print(cost_table)
                # There is no real protection here from reassigning
                # variable's tiling, outside of that variable's name
                # being present in assigned, we might want to add some
                # extra protection for that
                choices = [var.idx if assigned[var.name] else None for var in local_vars]
                reduction = 0
                for i in range(len(choices)):
                    if choices[i] is not None:
                        cost_table = cost_table.take(indices=choices[i], axis=i-reduction)
                        reduction += 1

                if not isinstance(cost_table, np.ndarray):
                    tmp_cost = cost_table
                else:
                    tmp_cost = cost_table.min()

                if tmp_cost < min_cost:
                    min_cost = tmp_cost
                    min_loc = np.unravel_index(cost_table.argmin(), cost_table.shape)
                    edge.set_idx_with_val(alg)
                    count = 0
                    for i in range(len(choices)):
                        if choices[i] is None:
                            var_stripped_idx = edge.vars.index(local_vars[i].name)
                            var_non_stripped_name = edge._vars[var_stripped_idx]
                            if var_non_stripped_name[-1] == 'T':
                                local_vars[i].idx = (min_loc[count]+1) % 2
                            else:
                                local_vars[i].idx = min_loc[count]
                            count += 1
            for var_name in edge.vars:
                assigned[var_name] = True
    return problem


def exhaust(problem, var_names, edge_names, size, verbosity=0, skip_real_exhaustive=False):
    vars_solution = [problem.variables[var_name] for var_name in var_names]
    edges_solution = [problem.edges[edge_name] for edge_name in edge_names]
    total_solution = vars_solution+edges_solution

    if skip_real_exhaustive:
        print("Size of space: ", size)

    num_cpus = cpu_count()
    num_cpus_used = floor(log(num_cpus, 2))

    # In case we have more cpus than iterations
    if num_cpus_used > size:
        num_cpus_used = floor(log(size, 2))

    problem_copies = []
    for i in range(num_cpus_used):
        problem_copies.append(copy.deepcopy(problem))

    def run_sub_exhaust(problem2, iter_idx):
        vars_solution2 = [problem2.variables[var_name] for var_name in var_names]
        edges_solution2 = [problem2.edges[edge_name] for edge_name in edge_names]
        total_solution2 = vars_solution2 + edges_solution2

        solution_map2 = {point.name: point.get_option() for point in total_solution}
        best_solution2 = solution_map2.copy()
        best_cost2 = problem.calculate_cost()

        # There shouldn't be any risk of truncation here
        # because of the floor used to calculate num_cpus_used
        num_vars_needed = int(log(num_cpus_used, 2))
        for idx in range(iter_idx):
            total_solution2[0].next(total_solution2[:num_vars_needed])

        finished = len(total_solution2) == 0
        while not finished:
            finished = total_solution2[num_vars_needed].next(total_solution2[num_vars_needed:])
            tmp_cost = problem.calculate_cost()
            if tmp_cost < best_cost2:
                best_cost2 = tmp_cost
                best_solution2 = {point.name: point.get_option() for point in total_solution2}
        return best_cost2, best_solution2

    pairs = Parallel(n_jobs=2)(delayed(run_sub_exhaust)(problem_copies[i], i) for i in range(len(problem_copies)))

    solution_map = {point.name: point.get_option() for point in total_solution}
    best_solution = solution_map.copy()
    best_cost = problem.calculate_cost()

    for i in range(len(pairs)):
        if pairs[i][0] < best_cost:
            best_cost = pairs[i][0]
            best_solution = pairs[i][1]

    for point in total_solution:
        point.set_idx_with_val(best_solution[point.name])
    return best_cost, best_solution


def greedy_solve_helper(problem, b, eta):
    edges_prime = set(problem.edges.keys())
    decided_tiling = set()
    num_vars = len(problem.variables)
    while len(decided_tiling) < num_vars:
        edge_bucket = compute_greedy_order(problem, edges_prime, decided_tiling, b, eta)
        t_prime = set()
        for edge_name in edge_bucket:
            for var_name in problem.edges[edge_name].vars:
                if var_name not in decided_tiling:
                    t_prime.add(var_name)
        exhaust(problem, t_prime, [], 3**len(t_prime))
        decided_tiling = decided_tiling | t_prime
        edges_prime -= edge_bucket
    return problem


def get_edge_min_cost(problem, edge_name, decided_tiling):
    edge = problem.edges[edge_name]
    finished = False
    tmp_min = sys.maxsize
    var_set = [problem.variables[var_name] for var_name in edge.vars if var_name not in decided_tiling]

    if len(var_set) > 0:
        while not finished:
            # We could use the cost table, and do an argmin op
            # but this is more resilient to how we get those costs
            # For example, if we change to an actual cost function
            finished = var_set[0].next(var_set)
            tmp_cost = problem.calculate_edge_subset_cost([edge.name])
            tmp_min = min(tmp_min, tmp_cost)
    else:
        return min(tmp_min, problem.calculate_edge_subset_cost([edge.name]))
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
    sorted_eps = list(gamma.items())
    sorted_eps = [(elem[1], elem[0]) for elem in sorted_eps]
    sorted_eps.sort(reverse=True)
    edges_double_prime = set()
    while len(edges_double_prime) < b and i < len(sorted_eps):
        if sorted_eps[i][0] >= eta*sorted_eps[0][0]:
            edges_double_prime.add(sorted_eps[i][1])
        else:
            return edges_double_prime
        i += 1
    return edges_double_prime
