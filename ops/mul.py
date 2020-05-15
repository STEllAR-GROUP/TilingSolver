import edge

from detail import MatrixSize


class Mul(edge.Edge):
    num_inputs = 2
    expression = "{} = {}*{}"
    op_name = "mul"

    def __init__(self, program_index, output, inputs):

        super(Mul, self).__init__(program_index, output, inputs)

    @staticmethod
    def output_size(operands):
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

    @staticmethod
    def valid_input_sizes():
        return [(MatrixSize.large_large, MatrixSize.large_large),
                (MatrixSize.small_small, MatrixSize.small_small),
                (MatrixSize.large_small, MatrixSize.small_large),
                (MatrixSize.small_large, MatrixSize.large_small)]

    @staticmethod
    def mul_cannon_cost(operands):
        assert len(operands) == 2, "Multiplication takes two arguments"
        assert operands[0] == 'block', "Cannon's algorithm requires block tiling"
        assert operands[0] == operands[1], "Cannon's algorithm requires block tiling"
        return 0

    @staticmethod
    def mul_dot_d_cost(operands):
        assert len(operands) == 2, "Multiplication takes two arguments"
        lhs = operands[0]
        rhs = operands[1]
        if lhs == 'row':
            if rhs == 'col':
                return 0
            # Not as much fetching if row-major on rhs
            elif rhs == 'row':
                return 3
            else:
                return 4
        elif lhs == 'col':
            if rhs == 'col' or rhs == 'block':
                return 2
            else:
                return 5
        else:  # lhs.tiling_type == 'block'
            if rhs == 'row':
                return 0
            elif rhs == 'block':
                return 3
            else:
                return 4

    @staticmethod
    def get_cost_dict():
        return {'cannon': Mul.mul_cannon_cost, 'dot_d': Mul.mul_dot_d_cost}
