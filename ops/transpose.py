import edge
import numpy as np

from matrix_size import MatrixSize


class Transpose(edge.Edge):
    num_inputs = 1
    expression = "{} = ({}).T"
    op_name = "transpose"
    _reassignable = False
    options = ['normal']

    def __init__(self, program_index, output, inputs):
        super(Transpose, self).__init__(program_index, output, inputs)

    @staticmethod
    def output_size(operands):
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

    @staticmethod
    def valid_input_sizes():
        return [(MatrixSize.large_large,),
                (MatrixSize.large_small,),
                (MatrixSize.small_large,),
                (MatrixSize.small_small,)]

    @staticmethod
    def normal_cost():
        return np.array([[5, 1, 2], [1.1, 5, 2], [3, 3, 1.1]])

    @staticmethod
    def get_cost_dict():
        return {'normal': Transpose.normal_cost}

    @staticmethod
    def num_implementations():
        return 1

    @staticmethod
    def random_imp():
        return "normal"

    def expression_weight(self):
        return 1.0

    def get_acceptable_tilings(self):
        r = "row"
        c = "col"
        b = "block"
        acceptable = [[r, r],
                      [c, c],
                      [b, b]]
        return acceptable
