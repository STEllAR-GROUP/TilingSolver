import edge
import numpy as np

from matrix_size import MatrixSize


class Add(edge.Edge):
    num_inputs = 2
    expression = "{} = {}+{}"
    op_name = "add"
    _reassignable = True
    options = ['normal']

    def __init__(self, program_index, output, inputs):
        super(Add, self).__init__(program_index, output, inputs)

    @staticmethod
    def output_size(operands):
        # lhs/rhs should be structured as (MatrixSize, distribution)
        assert len(operands) == 2, 'Matrix addition takes two arguments'
        lhs = operands[0]
        rhs = operands[1]
        assert lhs[0] == rhs[0], "Add operation should have equal size"
        assert lhs[1] == rhs[1], "Add operation should have aligned tiles"
        return lhs

    @staticmethod
    def valid_input_sizes():
        return [(MatrixSize.large_large, MatrixSize.large_large),
                (MatrixSize.large_small, MatrixSize.large_small),
                (MatrixSize.small_large, MatrixSize.small_large),
                (MatrixSize.small_small, MatrixSize.small_small)]

    @staticmethod
    def normal_add_cost():
        return np.array([[[1, 4, 2], [4, 4, 6], [2, 6, 2]],
                         [[4, 4, 6], [4, 1, 2], [6, 2, 2]],
                         [[2, 6, 2], [6, 2, 2], [2, 2, 1]]])

    @staticmethod
    def get_cost_dict():
        return {'normal': Add.normal_add_cost}

    @staticmethod
    def num_implementations():
        return 1

    @staticmethod
    def random_imp():
        return "normal"
