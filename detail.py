from matrix_size import MatrixSize
from ops.add import Add
from ops.inv import Inv
from ops.mul import Mul
from ops.transpose import Transpose


def get_edge_types():
    # This could be found with a sys call
    # to look at files in the ops folder
    # and importing them
    # So adding an op is easier, and we don't
    # have to maintain this list manually
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

