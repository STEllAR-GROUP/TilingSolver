import ops
import os


from matrix_size import MatrixSize
from ops.add import Add
from ops.inv import Inv
from ops.mul import Mul
from ops.transpose import Transpose


class EdgeSpace:
    def __init__(self):
        # This may be unnecessarily complex, because it's not a static
        # data fetch, so the commented line at the bottom may be preferable
        filenames = next(os.walk('ops'))[2]
        for name in filenames:
            assert '.py' in name
        file_things = [name[:-3] for name in filenames]
        capitalize_first = [name[0].upper() + name[1:] for name in file_things]
        self.edge_types = [getattr(ops, name) for name in capitalize_first]

def get_edge_types():
    # May prefer using the EdgeSpace, as it dynamically
    # finds all the relevant ops. But this still needs more
    # refactoring
    return [Add, Mul, Inv, Transpose]


def get_arity_op_dict(name):
    calcs = {}
    for typ in get_edge_types():
        calcs[typ.num_inputs] = {}
    for typ in get_edge_types():
        calcs[typ.num_inputs][typ.op_name] = getattr(typ, name)
    return calcs


def get_output_size_calculators():
    return get_arity_op_dict('output_size')


def get_valid_input_lists():
    return get_arity_op_dict('valid_input_sizes')


def get_op_dict(name):
    calcs = {}
    for typ in get_edge_types():
        calcs[typ.op_name] = getattr(typ, name)()
    return calcs


def get_all_cost_dicts():
    return get_op_dict('get_cost_dict')


def get_level_sets(partial_order):
    sets = []
    in_degrees = dict(partial_order.in_degree)
    count = 0
    while len(in_degrees.keys()) > 0:
        tmp = [x[0] for x in in_degrees.items() if x[1] == 0]
        neighbors = [partial_order.neighbors(y) for y in tmp]
        # This is a Set to make testing easier, mainly
        sets += [set(tmp)]
        for k in tmp:
            in_degrees.pop(k)
        for i in neighbors:
            for j in i:
                in_degrees[j] -= 1
        count += 1
    return sets

