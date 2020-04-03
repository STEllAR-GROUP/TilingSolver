import detail
import networkx as nx
import numpy as np
#TODO from cost_functions import get_cost_dict
import random
from cost import Cost
from cost_calculations import get_cost_dict
from detail import MatrixSize
from itertools import permutations
from networkx.algorithms import bipartite
from networkx import DiGraph


class Edge:
    """
    Class for binding together Edge (assignment and arithmetic operation)
    data in the tiling solver

    INPUT:

    - edge_name -- A string representation of the edge_name

    - op_name -- Name of the operation being performed

    - output -- Variable being assigned

    - inputs -- Inputs to the operation

    - expression -- AST representation for the assignment and operation
    """
    def __init__(self, edge_name, op_name, output, inputs, expression):
        # We only support expressions with one assigned variable for now
        self.edge_name = edge_name
        self.op_name = op_name
        self.output = output
        self.inputs = inputs
        self.vars = [self.output]+self.inputs
        self.expression = expression

    def get_arity(self):
        return len(self.inputs)


class Vertex:
    """
    Class for binding together Vertex (variable) data in the tiling solver

    INPUT:

    - var_name -- A string representation of the variable's name

    - size -- MatrixSize representation of the variable's (matrix or vector)
    size

    - tiling_type -- TilingType (row-major, column-major, or block)

    - dist -- List of localities this Vertex is distributed across
    In the case where data is not distributed across a proper subset of
    available localities, (that is, when all available localities are used
    in distributing this data structure) loc_list is simply empty
    """
    def __init__(self, var_name, size, tiling_type, dist):
        self.var_name = var_name
        self.size = size
        self.tiling_type = tiling_type
        self.dist = dist


class Locality:
    """
    Class for tracking a locality and its memory budget, and other data
    Not in use yet
    """
    def __init__(self, memory_budget, idx):
        self.idx = idx
        self.memory_budget = memory_budget

    def __lt__(self, other):
        return self.memory_budget < other.memory_budget

    def __le__(self, other):
        return self.memory_budget <= other.memory_budget

    def __eq__(self, other):
        return self.memory_budget == other.memory_budget


class Problem:
    """
    Highest level object for the tiling solver

    INPUT:
    
    - edge_set -- A dictionary containing tuples of (op_name,
    [inputs/outputs to operation] as sets, expression from AST)

    We use the list of inputs/outputs to feed into the construction
    of the hypergraph.

    -- NOTE -- 
    Order for the inputs/outputs is important in the edge set, as the order
    informs how the cost function calculates cost.
    However, since we don't allow any variable to be assigned more than
    once, the fact that the bipartite graph may not preserve the order of
    inputs/outputs indefinitely doesn't matter. Input matrices to an operation
    shouldn't be assigned later since they were assigned at initialization/being
    passed into the function. Output matrices shouldn't be assigned again either,
    obviously, meaning this set of inputs/outputs is guaranteed to be unique,
    and equality can be tested by vertex membership agnostic of the order.
    -- NOTE --

    - vertex_sizes -- A tuple of lists of variable_names.
    The tuple is structured as (big_big, big_small, small_big, small_small)

    - num_locs -- Number of localities to be used in computation

    - (Optional) initial_distribution -- Dictionary for sets of localities each
    beginning distributed matrix is spread across
    """
    def __init__(self, edge_set, vertex_sizes, num_locs,
                 initial_distribution=None):
        if initial_distribution is None:
            self.even_dist = True
        else:
            self.even_dist = False

        assert len(edge_set.values()) > 0
        for k in edge_set.values():
            assert len(list(k)) == 3, "Input data formatted incorrectly"

        self.num_locs = num_locs
        self.edges = {edge_name:
                      Edge(edge_name, op_name, k[0], k[1:], expression)
                      for edge_name, (op_name, k, expression)
                      in edge_set.items()}

        self.hypergraph = nx.Graph()
        self.init_hypergraph()

        # Make sure the big vertex list only includes vertices in the
        # edge set
        self.ground_set = {n for n, d in self.hypergraph.nodes(data=True)
                           if d['bipartite'] == 1}
        print(self.ground_set, vertex_sizes)
        assert self.ground_set == (self.ground_set | set(vertex_sizes[0] +
                                                         vertex_sizes[1] +
                                                         vertex_sizes[2] +
                                                         vertex_sizes[3]))
        this_var_is_output_in_this_edge_name = {p.output: p.edge_name
                                                for p in self.edges.values()}
        depends_on = {p.edge_name: set([]) for p in self.edges.values()}
        for p in self.edges.values():
            for i in p.inputs:
                try:
                    # This 'if' is checking if one of our inputs is the output
                    # to this expression, i.e., it's in a '+=' type scenario,
                    # which we don't allow.
                    # this_var_is_output_in_this_edge_name should throw a key
                    # error when an input is not on the LHS
                    # of any expression, meaning it's an original input. If
                    # this happens, it means this operation may be one of the
                    # first allowed to execute (when all inputs have this
                    # dependency
                    if this_var_is_output_in_this_edge_name[i] == p.edge_name:
                        raise ValueError("Variable redefinition not allowed")
                    depends_on[p.edge_name].add(
                        this_var_is_output_in_this_edge_name[i])
                except KeyError:
                    depends_on[p.edge_name].add('_begin_')
        self.partial_order = nx.DiGraph(depends_on).reverse()
        self.output_size_calculators = detail.size.get_output_size_calculators()
        self.cost_dict = detail.cost_calculations.get_cost_dict()
        self.vertices = {}
        self.init_vertices(vertex_sizes, initial_distribution)

    def init_vertices(self, vertex_sizes, initial_locality_distribution):
        for i in range(len(vertex_sizes)):
            for var in vertex_sizes[i]:
                if not self.even_dist and var in \
                        initial_locality_distribution.keys():
                    dist = initial_locality_distribution[var]
                else:
                    dist = []
                self.vertices[var] = Vertex(var, MatrixSize(i+1), 'row', dist)
        # The first level set is the artificial '_begin_' node
        for level_set in self.get_level_sets()[1:]:
            for edge_name in level_set:
                edge = self.edges[edge_name]
                output_size_func = self.get_output_size_calculator(edge)
                operands = [self.get_size_and_distribution(k)
                            for k in edge.inputs]
                out_size, out_dist = output_size_func(operands)
                self.vertices[edge.output] = Vertex(edge.output, out_size,
                                                    'row', out_dist)

    def init_hypergraph(self):
        bipartite_edge_set = []
        for p in self.edges.values():
            for var in p.vars:
                bipartite_edge_set += [(p.edge_name, var)]
        blocks = [p.vars for p in self.edges.values()]
        # Flatten blocks into set
        var_names = {i for k in blocks for i in k}
        outputs = [p.output for p in self.edges.values()]
        # Ensure that all elements are assigned to at most once
        # Assumes that all operations are assignments, which precludes
        # in-place operations
        assert len(set(outputs)) == len(outputs), \
            "Variable redefinition not allowed"
        self.hypergraph.add_nodes_from(self.edges.keys(), bipartite=0)
        self.hypergraph.add_nodes_from(var_names, bipartite=1)
        self.hypergraph.add_edges_from(bipartite_edge_set)

    def __call__(self, *args, **kwargs):
        return self.cost()

    def cost(self):
        cost = 0
        cost_dict = get_cost_dict()
        tmp_alg_choice = {}
        tmp_tiling_choice = {}
        recommend_retiling = {e: False for e in self.edges}
        retiling_diff = {e: 999 for e in self.edges}
        # TODO - Make this progress through level sets, choose
        # options optimally from the first, then impose tiling changes later
        level_sets = self.get_level_sets()

        first = True
        for set in level_sets[1:]:
            # What we might want to do is sort by operation type, so that, for example,
            # multiplication operations get first dibs on calling tiling for input matrices
            # ^^^ TODO ^^^
            # TODO - Try to set up detection on lower levels for an input value which hasn't
            # been used until then
            if first:
                assigned = {var.var_name: False for var in self.vertices.values()}
            for edge_name in set:
                e = self.edges[edge_name]
                tmp_dict = cost_dict[e.op_name]
                algs = list(tmp_dict.keys())
                ins = [self.vertices[x] for x in e.inputs.copy()]
                tiling_matches = self.get_tiling_tuples(len(ins))
                vals = np.zeros((len(algs), len(tiling_matches)))
                # TODO - This should be an actual look-up table for each
                # algorithm
                for i in range(len(algs)):
                    for j in range(len(tiling_matches)):
                        tmp_alg_choice[e.edge_name] = algs[i]
                        try:
                            vals[i, j] = tmp_dict[algs[i]](tiling_matches[j])
                        except AssertionError:
                            vals[i, j] = 9999999999
                val = vals.min()
                alg_idx, tile_idx = np.unravel_index(vals.argmin(), vals.shape)
                if first:
                    for i in range(len(e.inputs)):
                        if not assigned[ins[i].var_name]:
                            ins[i].tiling_type = tiling_matches[tile_idx][i]
                            assigned[ins[i].var_name] = True
                    tmp_alg_choice[e.edge_name] = algs[alg_idx]
                    tmp_tiling_choice[e.edge_name] = tiling_matches[tile_idx]
                    cost += val
                else:
                    parent_tiles = tuple([var.tiling_type for var in ins])
                    parent_tile_idx = tiling_matches.index(parent_tiles)
                    parent_tile_compliant_vals = vals[:, parent_tile_idx]
                    parent_val = parent_tile_compliant_vals.min()
                    parent_alg_idx = parent_tile_compliant_vals.argmin()
                    tmp_alg_choice[e.edge_name] = algs[parent_alg_idx]
                    tmp_tiling_choice[e.edge_name] = tiling_matches[parent_tile_idx]
                    if parent_val > val:
                        recommend_retiling[e.edge_name] = True
                        retiling_diff[e.edge_name] = val - parent_val
                    cost += parent_val
                first = False
        return Cost(tmp_alg_choice, tmp_tiling_choice, recommend_retiling, retiling_diff, cost)

    def get_output_size_calculator(self, edge):
        return self.output_size_calculators[edge.get_arity()][edge.op_name]

    def get_level_sets(self):
        sets = []
        in_degrees = dict(self.partial_order.in_degree)
        while len(in_degrees.keys()) > 0:
            tmp = [x[0] for x in in_degrees.items() if x[1] == 0]
            neighbors = [self.partial_order.neighbors(y) for y in tmp]
            sets += [tmp]
            for k in tmp:
                in_degrees.pop(k)
            for i in neighbors:
                for j in i:
                    in_degrees[j] -= 1
        return sets

    def get_size_and_distribution(self, var_name):
        var = self.vertices[var_name]
        return var.size, var.dist

    def get_distribution(self, var_name):
        return self.vertices[var_name].dist

    def get_tiling_tuples(self, size):
        return sorted(list(set(permutations(['row', 'row', 'col', 'col', 'block', 'block'], size))))
