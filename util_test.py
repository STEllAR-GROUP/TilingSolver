import itertools
import random
import time

from detail import EdgeSpace, get_level_sets
from matrix_size import MatrixSize
from ops.add import Add
from ops.inv import Inv
from ops.mul import Mul
from ops.transpose import Transpose
from problem import Problem
from solve import local_solve, greedy_solve


def make_basic_edge_set():
    edge_set = {Add(0, 'f', ['a', 'b']),
                Add(1, 'f', ['f', 'a']),
                Mul(2, 'g', ['b', 'c']),
                Add(3, 'h', ['e', 'g']),
                Mul(4, 'i', ['d', 'c'])}
    # [s_s, s_l, l_s, l_l]
    vertex_sizes = [['d'], ['c'], ['a', 'b'], ['e']]
    return edge_set, vertex_sizes


def make_basic_edge_set_add_transpose():
    edge_set = {Add(0, 'f', ['a', 'bT']),
                Add(1, 'f', ['f', 'a']),
                Mul(2, 'g', ['b', 'c']),
                Add(3, 'h', ['e', 'gT']),
                Mul(4, 'i', ['d', 'c'])}
    # [s_s, s_l, l_s, l_l]
    vertex_sizes = [['d'], ['a', 'c'], ['b'], ['e']]
    return edge_set, vertex_sizes

def make_three_level_edge_set():
    edge_set = {Mul(0, 'd', ['b', 'a']),
                Add(1, 'e', ['d', 'c']),
                Mul(2, 'f', ['e', 'b']),
                Transpose(3, 'g', ['f'])}
    # [s_s, s_l, l_s, l_l]
    vertex_sizes = [['c'], ['b'], ['a'], []]
    return edge_set, vertex_sizes


def make_multi_component_edge_set():
    edge_set = {Mul(0, 'd', ['b', 'a']),
                Add(2, 'e', ['d', 'c']),
                Mul(4, 'f', ['e', 'b']),
                Transpose(6, 'g', ['f']),
                Add(1, 'dd', ['bb', 'aa']),
                Mul(3, 'ee', ['dd', 'cc']),
                Add(5, 'ee', ['ee', 'bb']),
                Transpose(7, 'gg', ['ee'])}
    # [s_s, s_l, l_s, l_l]
    vertex_sizes = [['c'], ['b', 'aa', 'bb'], ['a'], ['cc']]
    return edge_set, vertex_sizes


def run_four_tests(edge_set, vertex_sizes, verbosity=0, prob=None, skip_real_exhaustive=False):
    if prob is None:
        prob = Problem(edge_set, vertex_sizes)

    start = time.perf_counter()
    cost, result = local_solve(prob)
    stop = time.perf_counter()
    print("1.")
    print("    Local solve results: ", cost, result)
    print(f"    Time for completion: {stop-start:0.4f}")
    print("----------------------------------------")

    implementation_space_size = 1
    for edge in prob.edges:
        implementation_space_size *= prob.edges[edge].num_implementations()
    tiling_space_size = 3 ** len(prob.ground_set)
    tau = implementation_space_size*tiling_space_size+1
    # Start with a fresh problem
    prob.reset_indices()
    start = time.perf_counter()
    results = greedy_solve(prob, tau, verbosity=verbosity, skip_real_exhaustive=skip_real_exhaustive)
    stop = time.perf_counter()
    print("2.")
    print("    Exhaustive search results: ", results)
    print(f"    Time for completion: {stop-start:0.4f}")
    print("----------------------------------------")

    # Start with a fresh problem
    prob.reset_indices()
    start = time.perf_counter()
    results = greedy_solve(prob, tau_prime=(implementation_space_size+1), verbosity=verbosity)
    stop = time.perf_counter()
    print("3.")
    print("    Implementation space search result: ", results)
    print(f"    Time for completion: {stop-start:0.4f}")
    print("----------------------------------------")

    # Start with a fresh problem
    prob.reset_indices()
    start = time.perf_counter()
    results = greedy_solve(prob, tau=2, tau_prime=2, verbosity=verbosity)
    stop = time.perf_counter()
    print("4.")
    print("    Min deviance search result: ", results)
    print(f"    Time for completion: {stop-start:0.4f}")
    print("----------------------------------------")


def find_output_size(vertex_sizes, op, inputs, es):
    operands = []
    for i in inputs:
        found = False
        for j in range(1, 5):
            if i in vertex_sizes[j-1]:
                operands.append((MatrixSize(j), []))
                found = True
        if not found:
            raise ValueError("Input not in vertex sizes")
    out_size = es.get_output_size_calculators()[len(inputs)][op.op_name](operands)
    return out_size


def find_valid_algorithms_and_inputs(vertex_sizes, es):
    valid_sizes = es.get_valid_input_lists()
    options = {i: {} for i in valid_sizes}
    return_ = []
    types = {typ.op_name: typ for typ in es.edge_types}
    for i in valid_sizes:
        for j in valid_sizes[i]:
            size_tuples = valid_sizes[i][j]()
            options_tmp = []
            for k in size_tuples:
                indices = []
                # i is synonymous with arity here
                for l in range(i):
                    indices.append(k[l].value-1)
                size_tuple_options = vertex_sizes[indices[0]]
                # Stack products
                for l in range(1, i):
                    size_tuple_options = itertools.product(size_tuple_options, vertex_sizes[indices[l]])
                    size_tuple_options = [(*a, b) for a, b in size_tuple_options]
                options_tmp += size_tuple_options
            if len(options_tmp) > 0:
                options[i][j] = options_tmp
                return_ += [(i, types[j], tuple(a)) for a in options_tmp]
    return return_


def generate_input_var_sizes(input_var_names):
    vertex_sizes = [[], [], [], []]
    input_vars = input_var_names.copy()
    for i in range(len(input_vars)):
        size = random.randint(0, 3)
        vertex_sizes[size].append(input_vars[i])
    return vertex_sizes


def generate_random_problem(my_seed, num_expressions=None, num_input_vars=None):
    if my_seed is None:
        my_seed = random.randint(0, 100)
    random.seed(my_seed)
    edge_space = EdgeSpace()
    print("Seed is {0}".format(my_seed))
    exp_idx_tracker = {i.op_name: 0 for i in edge_space.edge_types}
    if num_input_vars is None:
        num_input_vars = random.randint(1, 8)

    # This max(x , y) is to ensure that there is enough
    # space to use all input variables in at least one expression
    if num_expressions is None:
        num_expressions = max(num_input_vars, random.randint(7, 20))
    input_var_names = list(range(num_input_vars))
    input_var_names = [chr(97+i) for i in input_var_names]
    main_inputs = input_var_names
    use_all_inputs = input_var_names.copy()
    vertex_sizes = generate_input_var_sizes(input_var_names)
    beginning_vertex_sizes = vertex_sizes.copy()
    all_vars = input_var_names.copy()
    prev_layer_added = []
    edge_set = set()
    program_index = 0
    new_var_name = chr(97+num_input_vars)
    while num_expressions > 0:
        expression_layer_size = random.choices(list(range(1, 5)), [0.3, .3, .3, .1])[0]
        expression_layer_size = min(num_expressions, expression_layer_size)
        new_var_names = []
        new_sizes = [[], [], [], []]
        while expression_layer_size > 0:
            valid_algorithms_and_args = find_valid_algorithms_and_inputs(vertex_sizes, edge_space)
            if len(use_all_inputs) > 0:
                elem = random.choice(use_all_inputs)
                use_all_inputs.remove(elem)
            else:
                # This comes from the equation:
                # q*i+2*i*q = 1.0, which is based on the choice for
                # p = 2*q, so just created matrices are twice as likely
                # to be used as those which were instantiated previously
                i = len(all_vars)
                j = len(prev_layer_added)
                q = 1 / (2*i+j)
                p = 2*q
                var_selection_set = all_vars+prev_layer_added
                probabilities = [q for k in all_vars]+[p for k in prev_layer_added]
                elem = random.choices(var_selection_set, probabilities)[0]
            new_valid_algs_and_args = [(i, j, a) for i, j, a in valid_algorithms_and_args if elem in a]
            arity, op, inputs = random.choice(new_valid_algs_and_args)
            name = new_var_name
            exp_idx_tracker[op.op_name] += 1
            new_var_name = chr(ord(new_var_name) + 1)
            new_var_names.append(name)
            inputs = list(inputs)
            edge_set.add(op(program_index, name, inputs))
            expression_layer_size -= 1
            num_expressions -= 1
            out_size = find_output_size(vertex_sizes, op, inputs, edge_space)
            new_sizes[out_size[0].value-1].append(name)
            program_index += 1
        for k in range(4):
            vertex_sizes[k] += new_sizes[k]
        all_vars += new_var_names
        prev_layer_added = new_var_names
    return Problem(edge_set, beginning_vertex_sizes, 1), main_inputs


def generate_entire_program(inputs, problem):
    # TODO add some functionality for obtaining
    # level sets of variables from the edge graph
    small_dim = 100
    large_dim = 3000
    dimension_map = {MatrixSize.small_small: (small_dim, small_dim),
                     MatrixSize.small_large: (small_dim, large_dim),
                     MatrixSize.large_small: (large_dim, small_dim),
                     MatrixSize.large_large: (large_dim, large_dim)}
    program = ""
    for i in inputs:
        row_dim, col_dim = dimension_map[problem.variables[i].size]
        program += (str(i)+" = np.random("+str(row_dim)+", "+str(col_dim)+")\n")
    for i in get_level_sets(problem.partial_order)[1:]:
        while len(i) > 0:
            elem = random.choice(list(i))
            i.remove(elem)
            program += (str(problem.edges[elem])+'\n')
    return program
