import detail
import networkx as nx
import numpy as np

from detail import EdgeSpace
from edge import Edge
from itertools import permutations
from matrix_size import MatrixSize
from vertex import Vertex


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
    for use when initial_distribution, or Locality will be used.

    - (Optional) initial_distribution -- Dictionary for sets of localities each
    beginning distributed matrix is spread across
    """
    def __init__(self, edge_set, vertex_sizes, num_locs=1,
                 initial_distribution=None, hypergraph=None, partial_order=None, edges=None, variables=None):
        self.edgespace = EdgeSpace()
        if initial_distribution is None:
            self.even_dist = True
        else:
            self.even_dist = False

        if hypergraph is not None \
                and partial_order is not None \
                and variables is not None\
                and edges is not None:
            vertex_sizes = [[], [], [], []]
            for i in range(1, 5):
                vertex_sizes[i-1] = [k.name for k in variables.values() if k.size == MatrixSize(i)]
            edge_set = edges.values()
        assert len(edge_set) > 0
        for k in edge_set:
            assert isinstance(k, Edge), "Input data formatted incorrectly"

        self.num_locs = num_locs
        self.edges = {edge.name: edge
                      for edge in edge_set}
        self.hypergraph = nx.Graph()
        self.init_hypergraph(hypergraph=hypergraph)
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

        self.partial_order = nx.DiGraph()
        self.init_digraph(duplicate_outputs, partial_order=partial_order)
        self.output_size_calculators = self.edgespace.get_output_size_calculators()
        self.cost_dict = self.edgespace.get_all_cost_dicts()
        self.variables = {}
        self.init_variables(vertex_sizes, duplicate_outputs, initial_distribution, variables)

    def init_digraph(self, duplicate_outputs, partial_order=None):
        # Vertices of the digraph are the expressions in the user program
        if partial_order is not None:
            self.partial_order = partial_order
            return
        depends_on = {p.name: set([]) for p in self.edges.values()}
        for p in self.edges.values():
            for i in p.inputs:
                try:
                    # If a variable is never redeclared, this is easy
                    if len(duplicate_outputs[p.output]) > 1 and p.output == i:
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
                            idx = len(duplicate_outputs[i]) - 1
                        depends_on[p.name].add(
                            duplicate_outputs[i][idx].name
                        )
                    elif len(duplicate_outputs[i]) > 0:
                        depends_on[p.name].add(
                            duplicate_outputs[i][0].name)
                    else:
                        raise KeyError
                except KeyError:
                    depends_on[p.name].add('_begin_')
        self.partial_order = nx.DiGraph(depends_on).reverse()

    def init_variables(self, vertex_sizes, duplicate_outputs, initial_locality_distribution, variables=None):
        if variables is not None:
            self.variables = variables
            return
        # Initialize all of the variables in the input layer
        for i in range(len(vertex_sizes)):
            for var in vertex_sizes[i]:
                if not self.even_dist and var in \
                        initial_locality_distribution.keys():
                    dist = initial_locality_distribution[var]
                else:
                    dist = []
                self.variables[var] = Vertex(var, MatrixSize(i + 1), 0, 'row', dist)
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
                    self.variables[var_name] = Vertex(var_name, out_size, generation, 'row', out_dist)
                else:
                    if edge.output not in self.variables.keys():
                        self.variables[edge.output] = Vertex(edge.output, out_size,
                                                             0, 'row',
                                                             out_dist)

    def init_hypergraph(self, hypergraph=None):
        if hypergraph is not None:
            self.hypergraph = hypergraph
            return
        bipartite_edge_set = []
        for p in self.edges.values():
            for var in p.vars:
                bipartite_edge_set += [(p.name, var)]
        blocks = [p.vars for p in self.edges.values()]
        # Flatten blocks into set
        var_names = {i for k in blocks for i in k}
        self.hypergraph.add_nodes_from(self.edges.keys(), bipartite=0)
        self.hypergraph.add_nodes_from(var_names, bipartite=1)
        self.hypergraph.add_edges_from(bipartite_edge_set)

    def __call__(self, *args, **kwargs):
        return self.calculate_cost()

    def calculate_cost(self):
        return self.calculate_edge_subset_cost(self.edges.keys())

    def reset_indices(self):
        for edge in self.edges.values():
            edge.idx = 0
        for var in self.variables.values():
            var.idx = var.start_idx

    def calculate_edge_subset_cost(self, edge_subset):
        tmp_sum = 0
        for edge_name in edge_subset:
            edge = self.edges[edge_name]
            cost_dict = edge.get_cost_dict()
            func = cost_dict[edge.options[edge.idx]]
            cost_matrix = func()
            loc = np.array([self.variables[name].idx for name in edge.vars])
            cost = cost_matrix[tuple(loc.T)]
            tmp_sum += edge.loop_weight*cost
        return tmp_sum

    def get_output_size_calculator(self, edge):
        return self.output_size_calculators[edge.get_arity()][edge.op_name]

    def get_size_and_distribution(self, var_name):
        var = self.variables[var_name]
        return var.size, var.dist

    def get_distribution(self, var_name):
        return self.variables[var_name].dist

    def get_tiling_tuples(self, size):
        return sorted(list(set(permutations(['row', 'row', 'col', 'col', 'block', 'block'], size))))
