import numpy as np

from nextable import Nextable

from matrix_size import get_matrix_weight


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
        self._inputs = inputs
        self.inplace = False
        self.loop_weight = loop_weight
        if self.output in self.inputs:
            self.inplace = True
        self._vars = [self.output]+self._inputs
        self.check_arity()

    def get_arity(self):
        return len(self.inputs)

    def check_arity(self):
        assert len(self.inputs) == self.num_inputs

    @property
    def vars(self):
        return self._vars

    @property
    def inputs(self):
        return self._inputs

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

    def get_var_info(self, var_dict):
        indices = np.array([var_dict[name].idx for name in self.vars])
        matrix_sizes = [var_dict[name].size for name in self.vars]
        return indices, matrix_sizes, 0

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

    def expression_weight(self):
        # This method will return the weight associated
        # with this expression, dependant on the implementations
        # available
        raise NotImplementedError

    def get_acceptable_tilings(self):
        # This method will check if the tilings supplied are
        # allowed given the current implementation which is selected
        raise NotImplementedError

    @staticmethod
    def find_closest_tiling(as_is, acceptable):
        optimal = as_is
        if optimal in acceptable:
            return [False for _ in as_is]
        best_cost = 9999999.9
        for acc in acceptable:
            change_mask = [False if i == j else True for i, j in zip(as_is, acc)]
            change_sum = float(sum(change_mask))
            if change_sum < best_cost:
                optimal = change_mask
                best_cost = float(change_sum)
        return optimal
