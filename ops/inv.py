import edge

from detail import MatrixSize


class Inv(edge.Edge):
    num_inputs = 1
    expression = "{} = ({})^-1"
    op_name = "inv"

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
        return 0

    @staticmethod
    def get_cost_dict():
        return {'normal': Inv.normal_cost}

