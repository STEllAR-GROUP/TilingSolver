from matrix_size import MatrixSize

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


def inv_output_size(operands):
    assert len(operands) == 1, 'Matrix inverse takes one argument'
    arg_size = operands[0][0]
    if arg_size == MatrixSize.large_small or arg_size == MatrixSize.small_large:
        raise ValueError("Inverse can only be performed on square matrices")
    else:
        # Copy size and tiling from original
        return operands[0]


def transpose_output_size(operands):
    assert len(operands) == 1, 'Matrix transposition takes one arguments'
    arg_size = operands[0][0]
    out_size = None
    if arg_size == MatrixSize.large_large:
        out_size = MatrixSize.large_large
    elif arg_size == MatrixSize.large_small:
        out_size = MatrixSize.small_large
    elif arg_size == MatrixSize.small_large:
        out_size = MatrixSize.large_small
    elif arg_size == MatrixSize.small_small:
        out_size = MatrixSize.small_small
    # Copy tiling from LHS
    return out_size, operands[0][1]


def get_output_size_calculators():
    return {1: {'inv': inv_output_size, 'transpose': transpose_output_size},
            2: {'add': add_output_size, 'mul': mul_output_size}}
