import enum
import sage.all
#TODO from cost_functions import get_cost_dict
from sage.combinat.designs.incidence_structures import IncidenceStructure as IS
from sage.graphs.digraph import DiGraph

class MatrixSize(enum.Enum):
    small = 1
    medium = 2
    large = 3

class Problem:
    """
    Highest level object for the tiling solver

    INPUT:
    
    - EdgeSet -- A dictionary containing tuples of (op_name,
    [inputs/outputs to operation], expression from AST)

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

    - Vertices -- A list of variable_names.
    This is a list of variables that are large matrices, while the rest are
    assumed to be no bigger than medium.

    - PartialOrder -- A dictionary listing the edges (operations) which a
    particular edge depends on for execution
    """
    def __init__(self, edge_set, big_vertices):
        blocks = [k for (_,k,_) in edge_set.values()]
        firsts = [k_0[0] for k_0 in blocks]
        # Ensure that all elements are assigned to at most once
        # Assumes that all operations are assignments, which precludes
        # in-place operations
        assert(len(set(firsts)) == len(firsts))
        self.hypergraph = IS(blocks)
        # Make sure the big vertex list only includes vertices in the
        # edge set
        ground_set = set(self.hypergraph.ground_set())
        assert(ground_set == (ground_set|set(big_vertices)))
        self.edge_set = edge_set
        vertex_set = self.hypergraph.ground_set()
        self.vertices = { k:((MatrixSize.large,'row') if k in big_vertices 
                          else (MatrixSize.medium, 'row'))
                            for k in vertex_set}
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

def test():
    edge_set = {'add_0':('add', ['a','b','c'], 'row'),
                'mul_0':('mul', ['f','b','c'], 'row'),
                'mul_1':('mul', ['d','a','b'], 'row'),
                'add_1':('add', ['e','d','c'], 'row')}
    big_vertices = ['a','c']    
    return Problem(edge_set, big_vertices)
    




