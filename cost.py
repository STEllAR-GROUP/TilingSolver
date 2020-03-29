from matrix_size import MatrixSize

def add_cost(operands):
    assert len(operands) == 2, "Addition takes two arguments"
    if operands[0].tiling_type == 'row':
        if operands[1].tiling_type == 'row':
            return 0
        else:
            return 2
    elif operands[0].tiling_type == 'col':
        if operands[1].tiling_type == 'col':
            return 0
        else:
            return 2
    if operands[0].tiling_type == 'block':
        if operands[1].tiling_type == 'block':
            return 0
        else:
            return 2


def mul_cannon_cost(operands):
    assert len(operands) == 2, "Multiplication takes two arguments"
    assert operands[0].tiling_type == 'block', "Cannon's algorithm requires block tiling"
    assert operands[0].tiling_type == operands[1].tiling_type, "Cannon's algorithm requires block tiling"
    return 0


def mul_dot_d_cost(operands):
    assert len(operands) == 2, "Multiplication takes two arguments"
    lhs = operands[0]
    rhs = operands[1]
    if lhs.tiling_type == 'row':
        if rhs.tiling_type == 'col':
            return 0
        # Not as much fetching if row-major on rhs
        elif rhs.tiling_type == 'row':
            return 3
        else:
            return 4
    elif lhs.tiling_type == 'col':
        if rhs.tiling_type == 'col' or rhs.tiling_type == 'block':
            return 2
        else:
            return 5
    else:  # lhs.tiling_type == 'block'
        if rhs.tiling_type == 'row':
            return 0
        elif rhs.tiling_type == 'block':
            return 3
        else:
            return 4


def get_cost_dict():
    return {'mul': {'cannon': mul_cannon_cost, 'dot_d': mul_dot_d_cost},
            'add': {'normal_add': add_cost}}
