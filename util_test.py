from problem import Edge


def make_basic_edge_set():
    edge_set = {Edge('add', 0, 'f', ['a', 'b'], 'row'),
                Edge('add', 1, 'f', ['f', 'a'], 'row'),
                Edge('mul', 2, 'g', ['b', 'c'], 'row'),
                Edge('add', 3, 'h', ['e', 'g'], 'row'),
                Edge('mul', 4, 'i', ['d', 'c'], 'row')}
    # [l_l, l_s, s_l, s_s]
    vertex_sizes = [['d'], ['c'], ['a', 'b'], ['e']]
    return edge_set, vertex_sizes
