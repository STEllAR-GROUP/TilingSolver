import numpy as np

from nextable import Nextable


class Edge(Nextable):
    """
    Class for binding together Edge (assignment and arithmetic operation)
    data in the tiling solver

    INPUT:

    - op_name -- Name of the operation being performed

    - program_index -- Index of operation within program

    - output -- Variable being assigned

    - inputs -- Inputs to the operation

    - expression -- AST representation for the assignment and operation
    """
    num_inputs = -1
    expression = "Not Implemented"
    op_name = "No-Op"
    _reassignable = False

    def __init__(self, program_index, output, inputs, loop_weight=1.0):
        super(Edge, self).__init__(self.op_name + str(program_index))
        # We only support expressions with one assigned variable for now
        self.program_index = program_index
        self.output = output
        self.inputs = inputs
        self.inplace = False
        self.loop_weight = loop_weight
        if self.output in self.inputs:
            self.inplace = True
        self.vars = [self.output]+self.inputs
        self.check_arity()

    def get_arity(self):
        return len(self.inputs)

    def check_arity(self):
        assert len(self.inputs) == self.num_inputs

    def __repr__(self):
        return self.expression.format(self.output, *self.inputs)

    def __str__(self):
        return self.expression.format(self.output, *self.inputs)

    def __lt__(self, other):
        return self.program_index < other.program_index

    def __le__(self, other):
        return self.program_index <= other.program_index

    def reassignable(self):
        return self._reassignable

    def set_min_cost_deviance_algorithm(self):
        costs = [dict_func() for dict_func in self.get_cost_dict().values()]
        deviations = [mat.max()/mat.min() for mat in costs]
        self.idx = np.argmin(deviations)

    @property
    def algorithm(self):
        return self.options[self.idx]

    @staticmethod
    def output_size(operands):
        raise NotImplementedError

    @staticmethod
    def valid_input_sizes():
        raise NotImplementedError

    @staticmethod
    def get_cost_dict():
        raise NotImplementedError

    @staticmethod
    def num_implementations():
        raise NotImplementedError

    @staticmethod
    def random_imp():
        raise NotImplementedError
