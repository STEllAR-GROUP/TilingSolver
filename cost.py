from matrix_size import MatrixSize

def add_map_cost(operands):
    size_mul = 1
    if operands[0][0] == MatrixSize.small_small:
        pass
    elif operands[0][0] == MatrixSize.large_large:
        size_mul = 4
    else:
        size_mul = 2
    if operands[0][0]:
        return


def add_other_cost(operands):
    if operands[0][0] == MatrixSize.small_small:
        return 0
    else:
        return 1


def mul_cannon_cost(operands):
    if operands[0][0] == MatrixSize.small_small:
        return 0
    else:
        return 1


def mul_dot_d_cost(operands):
    if operands[0][0] == MatrixSize.small_large:
        return 1
    else:
        return 2


def get_cost_dict():
    return {'mul': [mul_cannon_cost, mul_dot_d_cost],
            'add': [add_map_cost, add_other_cost]}