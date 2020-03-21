import enum


class MatrixSize(enum.Enum):
    small_small = 1
    small_large = 2
    large_small = 3
    large_large = 4


def add_output_size(operands):
    assert len(operands) == 2, 'Matrix addition takes two arguments'
    lhs = operands[0]
    rhs = operands[1]
    assert lhs[0] == rhs[0], "Add operation should have equal size"
    assert lhs[1] == rhs[1], "Add operation should have aligned tiles"
    return lhs


def mul_output_size(operands):
    assert len(operands) == 2, 'Matrix multiplication takes two arguments'
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
        return out_size, operands[0][1]


def get_output_size_calculators():
    return {2: {'add': add_output_size, 'mul': mul_output_size}}


def add_map_cost(operands):
    if operands[0][0] == MatrixSize.small_small:
        return 0
    else:
        return 1


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




















