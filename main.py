import networkx as nx
#TODO from cost_functions import get_cost_dict
from detail import get_output_size_calculators, get_cost_dict, MatrixSize
from networkx.algorithms import bipartite
from networkx import DiGraph

class Edge:
    """

    """
    def __init__(self, edge_name, op_name, output, inputs, expression):
        # We only support expressions with one assigned variable for now
        self.edge_name = edge_name
        self.op_name = op_name
        self.output = output
        self.inputs = inputs
        self.vars = self.output+self.inputs
        self.expression = expression

    def get_arity(self):
        return len(self.inputs)

class Vertex:
    """
    Even distribution across all available localities is encoded by an
    empty list
    """
    def __init__(self, var_name, size, tiling_type, loc_list):
        self.var_name = var_name
        self.size = size
        self.tiling_type = tiling_type
        self.loc_list = loc_list

class Locality:
    """
    Class for tracking a locality and its memory budget, and other data
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
    once, the fact that the IncidenceStructure modifies the order of the lists
    of inputs/outputs doesn't matter. Input matrices to an operation shouldn't
    be assigned later since they were assigned at initialization/being passed 
    into the function. Output matrices shouldn't be assigned again either, 
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
                       memory_budget=None, initial_distribution=None):
        self.num_locs = num_locs
        if(initial_distribution is None):
            self.even_dist = True
            self.locality_distribution = {}
        else:
            self.even_dist = False
            self.locality_distribution = initial_distribution
        self.vertex_sizes = vertex_sizes
        assert len(edge_set.values()) > 0
        assert len(list(edge_set.values())[0]) == 3
        self.edges = {edge_name: Edge(edge_name, op_name, k[:1], k[1:], expression)
                 for edge_name, (op_name, k, expression) in edge_set.items()}
        bipartite_edge_set = []
        for edge_name, variables in [(p.edge_name, p.vars) for p in self.edges.values()]:
            print(edge_name, variables)
            for i in variables:
                print(edge_name, i)
                bipartite_edge_set += [(edge_name, i)]
        blocks = [k for (_, k, _) in edge_set.values()]
        var_names = [i for k in blocks for i in k]
        firsts = [k_0[0] for (_, k_0, _) in edge_set.values()]
        # Ensure that all elements are assigned to at most once
        # Assumes that all operations are assignments, which precludes
        # in-place operations
        assert len(set(firsts)) == len(firsts)
        self.hypergraph = nx.Graph()
        self.hypergraph.add_nodes_from(edge_set.keys(), bipartite=0)
        self.hypergraph.add_nodes_from(var_names, bipartite=1)
        self.hypergraph.add_edges_from(bipartite_edge_set)

        # Make sure the big vertex list only includes vertices in the
        # edge set
        self.ground_set = {n for n, d in self.hypergraph.nodes(data=True) if d['bipartite'] == 1}
        assert self.ground_set == (self.ground_set | set(vertex_sizes[0] +
                                                         vertex_sizes[1] +
                                                         vertex_sizes[2] +
                                                         vertex_sizes[3]))
        self.edge_set = edge_set
        lhs = {l[0]: edge for (edge, (_, l, _)) in self.edge_set.items()}
        partial_order = {k: set([]) for k in self.edge_set.keys()}
        for name, (_, var_s, _) in self.edge_set.items():
            for i in var_s[1:]:
                try:
                    if lhs[i] != name:
                        partial_order[name].add(lhs[i])
                except KeyError:
                    partial_order[name].add('_begin_')
        self.partial_order = nx.DiGraph(partial_order).reverse()
        self.output_size_calculators = get_output_size_calculators()
        self.cost_dict = get_cost_dict()
        self.vertices = {}
        self.init_vertices()

    def get_output_size_calculator(self, edge):
        return self.output_size_calculators[edge.get_arity()][edge.op_name]

    def get_level_sets(self):
        queue = list(self.partial_order.nodes)
        sets = []
        in_deg = dict(self.partial_order.in_degree)
        while len(in_deg.keys()) > 0:
            tmp = [x[0] for x in in_deg.items() if x[1] == 0]
            neighbors = [self.partial_order.neighbors(y) for y in tmp]
            sets += [tmp]
            for k in tmp:
                in_deg.pop(k)
            for i in neighbors:
                for j in i:
                    in_deg[j] -= 1
        return sets

    def get_size_and_distribution(self, k):
        if not self.even_dist:
            if k in self.locality_distribution.keys():
                dist = self.locality_distribution[k]
        else:
            dist = None
        if k in self.vertex_sizes[0]:
            size = MatrixSize.large_large
        elif k in self.vertex_sizes[1]:
            size = MatrixSize.large_small
        elif k in self.vertex_sizes[2]:
            size = MatrixSize.small_large
        elif k in self.vertex_sizes[3]:
            size = MatrixSize.small_small
        else:
            raise ValueError('Error retrieving matrix size')
        return size, dist

    def set_size_and_distribution(self, size, distribution, var_name):
        if distribution is not None and not self.even_dist:
            self.locality_distribution[var_name] = distribution
        if size == MatrixSize.large_large and var_name not in self.vertex_sizes[0]:
            self.vertex_sizes[0] += var_name
        elif size == MatrixSize.large_small and var_name not in self.vertex_sizes[1]:
            self.vertex_sizes[1] += var_name
        elif size == MatrixSize.small_large and var_name not in self.vertex_sizes[2]:
            self.vertex_sizes[2] += var_name
        elif size == MatrixSize.small_small and var_name not in self.vertex_sizes[3]:
            self.vertex_sizes[3] += var_name
        else:
            raise ValueError('Variable found on LHS of multiple expressions')

    def init_vertices(self):
        # The first level set is the artificial '_begin_' node
        for level_set in self.get_level_sets()[1:]:
            for edge in level_set:
                tmp_edge = self.edges[edge]
                output_size_func = self.get_output_size_calculator(tmp_edge)
                operands = [self.get_size_and_distribution(k) for k in tmp_edge.inputs]
                out_size, out_dist = output_size_func(operands)
                self.set_size_and_distribution(out_size, out_dist, tmp_edge.output)

        for v in self.ground_set:
            size = self.get_size_and_distribution(v)
            if self.even_dist:
                self.vertices[v] = Vertex(v, size, 'row', [])
            else:
                self.vertices[v] = Vertex(v, size, 'row', [-1])  


def test():
    edge_set = {'add_0': ('add', ['f', 'a', 'b'], 'row'),
                'mul_0': ('mul', ['g', 'b', 'c'], 'row'),
                'mul_1': ('add', ['h', 'e', 'g'], 'row'),
                'add_1': ('mul', ['i', 'd', 'c'], 'row')}
    # [l_l, l_s, s_l, s_s]
    vertex_sizes = [['e'], ['a', 'b'], ['c'], ['d']]
    return Problem(edge_set, vertex_sizes, 1)
    




