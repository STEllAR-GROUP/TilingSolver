from matrix_size import MatrixSize

def add_map_tiling(operands):
    """
    Returns cost proper tiling for a given set of input tilings.

    In the case of the ordinary addition operation, if both inputs have the
    same tiling, we return that tiling scheme. If not, if either tiling is
    row-major we use row. Otherwise, we default to the tiling of the LHS.

    INPUT:

    - operands -- List of vertex objects as input

    RETURNS:

    - Tiling -- String representation of the output tiling
    """
    assert len(operands) == 2, "Addition takes two arguments only"
    assert operands[0].size == operands[1].size, "Addition of matrices requires equal dimensions"
    if operands[0].tiling_type == operands[1].tiling_type:
        return operands[0].tiling_type
    elif operands[0].tiling_type == "row" or operands[1].tiling_type == "row":
        return "row"
    else:
        return operands[0].tiling_type


def mul_cannon_tiling(operands):
    assert len(operands) == 2, "Matrix multiplication requires two arguments"
    assert operands[0].tiling_type == "block", "Cannon's algorithm only allows block tiling"
    assert operands[0].tiling_type == operands[1].tiling_type, "Cannon's algorithm only allows block tiling"

    return "block"


def mul_dot_d_tiling(operands):
    assert len(operands) == 2, "Matrix multiplication requires two arguments"
    return operands[0].tiling_type


def get_tiling_dict():
    return {'mul': [mul_cannon_tiling, mul_dot_d_tiling],
            'add': [add_map_tiling]}
