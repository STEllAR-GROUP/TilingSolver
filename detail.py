from matrix_size import MatrixSize
import cost_calculations
import tiling
import size


def get_level_sets(partial_order):
    sets = []
    in_degrees = dict(partial_order.in_degree)
    count = 0
    while len(in_degrees.keys()) > 0:
        tmp = [x[0] for x in in_degrees.items() if x[1] == 0]
        neighbors = [partial_order.neighbors(y) for y in tmp]
        # This is a set to make testing easier, mainly
        sets += [set(tmp)]
        for k in tmp:
            in_degrees.pop(k)
        for i in neighbors:
            for j in i:
                in_degrees[j] -= 1
        count += 1
    return sets




















