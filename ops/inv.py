import edge
import numpy as np

from matrix_size import MatrixSize


class Inv(edge.Edge):
    num_inputs = 1
    expression = "{} = ({})^-1"
    op_name = "inv"
    _reassignable = False
    options = ['normal']

    def __init__(self, program_index, output, inputs):
        super(Inv, self).__init__(program_index, output, inputs)

    @staticmethod
    def output_size(operands):
        assert len(operands) == 1, 'Matrix inverse takes one argument'
        arg_size = operands[0][0]
        if arg_size == MatrixSize.large_small or arg_size == MatrixSize.small_large:
            raise ValueError("Inverse can only be performed on square matrices")
        else:
            # Copy size and tiling from original
            return operands[0]

    @staticmethod
    def valid_input_sizes():
        return [(MatrixSize.large_large,),
                (MatrixSize.small_small,)]

    @staticmethod
    def normal_cost():
        return np.array([[1, 10, 2], [6, 10, 4], [2, 3, 1.1]])

    @staticmethod
    def get_cost_dict():
        return {'normal': Inv.normal_cost}

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

