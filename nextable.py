

class Nextable:
    def __init__(self):
        self.i = 1

    def num_iterations(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def iterate_over_lower(self, problem, sub_problem):
        # Basically this method allows any node (vertex or edge) to
        # participate in the iteration over all possible solutions
        # to the tiling problem
        # This
        my_idx = sub_problem.index(self)
