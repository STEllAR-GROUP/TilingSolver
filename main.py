import enum
import sage.all
#TODO from cost_functions import get_cost_dict
from detail import get_output_size_calculators, get_cost_dict, MatrixSize
from sage.combinat.designs.incidence_structures import IncidenceStructure as IS
from sage.graphs.digraph import DiGraph

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
        else:
            self.even_dist = False
            self.initial_distribution = initial_distribution
        self.vertex_sizes = vertex_sizes
        assert len(edge_set.values()) > 0
        assert len(list(edge_set.values())[0]) == 3
        blocks = [k for (_,k,_) in edge_set.values()]
        firsts = [k_0[0] for k_0 in blocks]
        # Ensure that all elements are assigned to at most once
        # Assumes that all operations are assignments, which precludes
        # in-place operations
        assert len(set(firsts)) == len(firsts)
        self.hypergraph = IS(blocks)
        # Make sure the big vertex list only includes vertices in the
        # edge set
        ground_set = set(self.hypergraph.ground_set())
        assert ground_set == (ground_set|set(vertex_sizes[0]
                                            +vertex_sizes[1]
                                            +vertex_sizes[2]
                                            +vertex_sizes[3]))
        self.edge_set = edge_set
        lhs = { l[0] : edge for (edge, (_,l,_)) in self.edge_set.items()}
        partial_order = { k: set([]) for k in self.edge_set.keys() }
        for name, (_,var_s,_) in self.edge_set.items():
            for i in var_s[1:]:
                try:
                    if lhs[i] != name:
                        partial_order[name].add(lhs[i])
                except KeyError:
                    partial_order[name].add('_begin_')
        self.partial_order = DiGraph(partial_order).reverse()
        vertex_set = self.hypergraph.ground_set()
        
        self.output_size_calculators = get_output_size_calculators()
        self.cost_dict = get_cost_dict()
        self.vertices = {}
        self.init_vertices(vertex_set)



    def init_vertices(self, vertex_set):
        print("Inside init_vertices")
        for level_set in self.partial_order.level_sets()[1:]:
            for edge in level_set:
                op, vars_, expr = self.edge_set[edge]
                output_size_func = self.output_size_calculators[
                    len(vars_[1:])][op]
                operands = []
                for k in vars_[1:]:
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
                    if not self.even_dist:
                        if k in initial_distribution.keys():
                            operands.append((size, initial_distribution[k]))
                    else:
                        operands.append((size, None))
                out_size = output_size_func(operands)
                if out_size[0] == MatrixSize.large_large:
                    self.vertex_sizes[0].append(vars_[0])
                elif out_size[0] == MatrixSize.large_small:
                    self.vertex_sizes[1].append(vars_[0])
                elif out_size[0] == MatrixSize.small_large:
                    self.vertex_sizes[2].append(vars_[0])
                elif out_size[0] == MatrixSize.small_small:
                    self.vertex_sizes[3].append(vars_[0])
        for v in vertex_set:
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
            if self.even_dist:
                self.vertices[v] = Vertex(v, size, 'row', [])
            else:
                self.vertices[v] = Vertex(v, size, 'row', [-1])  


                

def test():
    edge_set = {'add_0':('add', ['f','a','b'], 'row'),
                'mul_0':('mul', ['g','b','c'], 'row'),
                'mul_1':('add', ['h','e','g'], 'row'),
                'add_1':('mul', ['i','d','c'], 'row')}
    # [l_l, l_s, s_l, s_s]
    big_vertices = [['e'],['a','b'],['c'],['d']]    
    return Problem(edge_set, big_vertices, 1)
    




