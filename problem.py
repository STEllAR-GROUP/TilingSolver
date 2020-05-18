import detail
import networkx as nx
import numpy as np
#TODO from cost_functions import get_cost_dict
import random

from cost import Cost
from detail import MatrixSize, get_all_cost_dicts
from edge import Edge
from itertools import permutations
from networkx.algorithms import bipartite
from networkx import DiGraph


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
    tiling_types = ['row', 'col', 'block']

    def __init__(self, var_name, size, generation, tiling_type, dist):
        # Add generation value to track how many times this variable changes
        self.var_name = var_name
        self.size = size
        # The combination of the var_name and generation should give a
        # globally unique identifier to the data enclosed therein.
        # This is mainly because the generation can't be the same for
        # two different reassignments (it might be if we genuinely parsed
        # conditional structures) due to how the dependency graph calculation
        # works
        self.generation = generation
        assert tiling_type in self.tiling_types
        self.tiling_type = tiling_type
        self.dist = dist
        self.tiling_idx = self.tiling_types.index(self.tiling_type)

    def next_tiling(self):
        self.tiling_idx = (self.tiling_idx+1) % len(self.tiling_types)
        self.tiling_type = self.tiling_types[self.tiling_idx]


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
    
    - edge_set -- A set containing Edge objects

    - vertex_sizes -- A list of lists of variable_names.
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

        assert len(edge_set) > 0
        for k in edge_set:
            assert isinstance(k, Edge), "Input data formatted incorrectly"

        self.num_locs = num_locs
        self.edges = {edge.edge_name: edge
                      for edge in edge_set}
        self.hypergraph = nx.Graph()
        self.init_hypergraph()
        # Make sure the big vertex list only includes vertices in the
        # edge set
        self.ground_set = {n for n, d in self.hypergraph.nodes(data=True)
                           if d['bipartite'] == 1}
        assert self.ground_set == (self.ground_set | set(vertex_sizes[0] +
                                                         vertex_sizes[1] +
                                                         vertex_sizes[2] +
                                                         vertex_sizes[3]))

        unique_output_list = set([p.output for p in self.edges.values()])
        duplicate_outputs = {p: [] for p in unique_output_list}
        for edge in self.edges.values():
            duplicate_outputs[edge.output] += [edge]
        for p in duplicate_outputs:
            duplicate_outputs[p] = sorted(duplicate_outputs[p])

        depends_on = {p.edge_name: set([]) for p in self.edges.values()}
        for p in self.edges.values():
            for i in p.inputs:
                try:
                    # If a variable is never redeclared, this is easy
                    if len(duplicate_outputs[i]) > 1:
                        # Find in the duplicate outputs the one that has program index less than
                        # i, but is the maximal one where that condition holds
                        # We want to ensure we allow redeclaration for this op
                        # For instance, redeclaration of multiplication doesn't make sense,
                        # cause we have to allocate memory for an intermediate matrix anyway,
                        # so we might as well just mark it as a new var entirely
                        # But if we're doing it in place like negation, or a +=, we don't
                        # need an intermediate matrix
                        assert p.reassignable(), "Operation {} not in-place reassign-able".format(p.__str__())
                        idx = 0
                        for j in duplicate_outputs[i]:
                            if p.program_index > j.program_index:
                                break
                            idx += 1
                        if idx == len(duplicate_outputs[i]):
                            idx = len(duplicate_outputs[i])-1
                        depends_on[p.edge_name].add(
                            duplicate_outputs[i][idx].edge_name
                        )
                    elif len(duplicate_outputs[i]) > 0:
                        depends_on[p.edge_name].add(
                            duplicate_outputs[i][0].edge_name)
                    else:
                        raise KeyError
                except KeyError:
                    depends_on[p.edge_name].add('_begin_')
        self.partial_order = nx.DiGraph(depends_on).reverse()
        self.output_size_calculators = detail.get_output_size_calculators()
        self.cost_dict = detail.get_all_cost_dicts()
        self.vertices = {}
        self.init_vertices(vertex_sizes, duplicate_outputs, initial_distribution)

    def init_vertices(self, vertex_sizes, duplicate_outputs, initial_locality_distribution):
        # Initialize all of the variables in the input layer
        for i in range(len(vertex_sizes)):
            for var in vertex_sizes[i]:
                if not self.even_dist and var in \
                        initial_locality_distribution.keys():
                    dist = initial_locality_distribution[var]
                else:
                    dist = []
                self.vertices[var] = Vertex(var, MatrixSize(i+1), 0, 'row', dist)
        # The first level set is the artificial '_begin_' node
        for level_set in detail.get_level_sets(self.partial_order)[1:]:
            for edge_name in level_set:
                edge = self.edges[edge_name]
                output_size_func = self.get_output_size_calculator(edge)
                operands = [self.get_size_and_distribution(k)
                            for k in edge.inputs]
                generation = duplicate_outputs[edge.output].index(edge)
                # TODO - Might be worthwhile to verify that this output
                # only happens once in this level_set as output,
                # otherwise we're doing something wrong in the dependency
                # graph generation. But maybe that should just be a test?
                out_size, out_dist = output_size_func(operands)
                var_name = edge.output+str(generation)
                # This is a temporary measure. We might want to do more nuanced
                # analyses where we discriminate between reassignments. If so,
                # this should be a higher level config setting
                use_duplicates = False
                if use_duplicates:
                    self.vertices[var_name] = Vertex(var_name, out_size, generation, 'row', out_dist)
                else:
                    if edge.output not in self.vertices.keys():
                        self.vertices[edge.output] = Vertex(edge.output, out_size,
                                                            0, 'row',
                                                            out_dist)

    def init_hypergraph(self):
        bipartite_edge_set = []
        for p in self.edges.values():
            for var in p.vars:
                bipartite_edge_set += [(p.edge_name, var)]
        blocks = [p.vars for p in self.edges.values()]
        # Flatten blocks into set
        var_names = {i for k in blocks for i in k}
        self.hypergraph.add_nodes_from(self.edges.keys(), bipartite=0)
        self.hypergraph.add_nodes_from(var_names, bipartite=1)
        self.hypergraph.add_edges_from(bipartite_edge_set)

    def __call__(self, *args, **kwargs):
        return self.cost()

    def cost(self):
        cost = 0
        cost_dict = get_all_cost_dicts()
        tmp_alg_choice = {}
        tmp_tiling_choice = {}
        recommend_retiling = {e: False for e in self.edges}
        retiling_diff = {e: 999 for e in self.edges}
        # TODO - Make this progress through level sets, choose
        # options optimally from the first, then impose tiling changes later
        level_sets = detail.get_level_sets(self.partial_order)

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

    def get_size_and_distribution(self, var_name):
        var = self.vertices[var_name]
        return var.size, var.dist

    def get_distribution(self, var_name):
        return self.vertices[var_name].dist

    def get_tiling_tuples(self, size):
        return sorted(list(set(permutations(['row', 'row', 'col', 'col', 'block', 'block'], size))))
