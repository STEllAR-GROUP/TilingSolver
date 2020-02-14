import enum
import sage.all
#TODO from cost_functions import get_cost_dict
from sage.combinat.designs.incidence_structures import IncidenceStructure as IS
from sage.graphs.digraph import DiGraph

class MatrixSize(enum.Enum):
    small_small = 1
    small_large = 2
    large_small = 3
    large_large = 4

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
    def __init__(self, edge_set, vertex_sizes, num_locs, initial_distribution=None):
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
        
        self.output_size_calculators = {2:{'add':self.add_output_size, 'mul':self.mul_output_size}}
        self.vertices = {}
        self.init_vertices(vertex_set)



    def init_vertices(self, vertex_set):
        print("Inside init_vertices")
        for level_set in self.partial_order.level_sets()[1:]:
            for edge in level_set:
                op, vars_, expr = self.edge_set[edge]
                output_size_func = self.output_size_calculators[len(vars_[1:])][op]
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
                


    def add_output_size(self, operands):
        lhs = operands[0]
        rhs = operands[1]
        assert lhs[0] == rhs[0], "Add operation should have equal size"
        assert lhs[1] == rhs[1], "Add operation should have aligned tiles"
        return lhs


    def mul_output_size(self, operands):
        assert len(operands), 'Matrix multiplication takes two arguments'
        lhs_size = operands[0][0]
        rhs_size = operands[1][0]
        out_size = None
        if lhs_size == MatrixSize.large_large:
            if rhs_size == MatrixSize.large_large:
                out_size = MatrixSize.large_large
            elif rhs_size == MatrixSize.large_small:
                out_size = MatrixSize.large_small
        elif lhs_size == MatrixSize.large_small:
            if rhs_size == MatrixSize.small_large:
                out_size = MatrixSize.large_large
            elif rhs_size == MatrixSize.small_small:
                out_size = MatrixSize.large_small
        elif lhs_size == MatrixSize.small_large:
            if rhs_size == MatrixSize.large_large:
                out_size = MatrixSize.small_large
            elif rhs_size == MatrixSize.large_small:
                out_size = MatrixSize.small_small
        elif lhs_size == MatrixSize.small_small:
            if rhs_size == MatrixSize.small_large:
                out_size = MatrixSize.small_large
            elif rhs_size == MatrixSize.small_small:
                out_size = MatrixSize.small_small
        if out_size is None:
            raise ValueError('Matrix size mismatch {0}, {1}'.format(lhs_size, rhs_size))
        else:
            # Copy tiling from LHS
            return (out_size, operands[0][1])    

def test():
    edge_set = {'add_0':('add', ['f','a','b'], 'row'),
                'mul_0':('mul', ['g','b','c'], 'row'),
                'mul_1':('add', ['h','e','g'], 'row'),
                'add_1':('mul', ['i','d','c'], 'row')}
    # [l_l, l_s, s_l, s_s]
    big_vertices = [['e'],['a','b'],['c'],['d']]    
    return Problem(edge_set, big_vertices, 1)
    




