import edge

from ops.add import Add
from ops.inv import Inv
from ops.mul import Mul
from ops.transpose import Transpose



def make_basic_edge_set():
    edge_set = {Add(0, 'f', ['a', 'b']),
                Add(1, 'f', ['f', 'a']),
                Mul(2, 'g', ['b', 'c']),
                Add(3, 'h', ['e', 'g']),
                Mul(4, 'i', ['d', 'c'])}
    # [l_l, l_s, s_l, s_s]
    vertex_sizes = [['d'], ['c'], ['a', 'b'], ['e']]
    return edge_set, vertex_sizes
