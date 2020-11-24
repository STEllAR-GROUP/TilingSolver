import edge
from ops.transpose import Transpose
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

    def get_var_info(self, var_dict):
        stripped_vars = self.vars
        flip = [x != y for (x, y) in zip(stripped_vars, self._vars)]
        loc = np.zeros(len(stripped_vars), dtype=np.int32)
        matrix_sizes = []
        mod_cost = 0
        transpose_cost_matrix = Transpose.normal_cost()
        for i in range(loc.shape[0]):
            var = var_dict[stripped_vars[i]]
            matrix_sizes.append(var.size)
            if flip[i]:
                loc[i] = var.get_opposite_idx()
                mod_cost += transpose_cost_matrix[loc[i], var.idx]
            else:
                loc[i] = var.idx
        return loc, matrix_sizes, mod_cost

    @property
    def vars(self):
        stripped_vars = [i[:-1] if i[-1] == 'T' else i for i in self._vars]
        return stripped_vars

    @property
    def inputs(self):
        return [i[:-1] if i[-1] == 'T' else i for i in self._inputs]

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
                         [[4, 4, 6], [4, 1.1, 2], [6, 2, 2]],
                         [[2, 6, 2], [6, 2, 2], [2, 2, 1.1]]])

    @staticmethod
    def get_cost_dict():
        return {'normal': Add.normal_add_cost}

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
        acceptable = [[r, r, r],
                      [c, c, c],
                      [b, b, b]]
        return acceptable
