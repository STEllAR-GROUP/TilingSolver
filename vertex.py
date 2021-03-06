from nextable import Nextable
from matrix_size import MatrixSize


class Vertex(Nextable):
    """
    Class for binding together Vertex (variable) data in the tiling solver

    INPUT:

    - var_name -- A string representation of the variable's name

    - size -- MatrixSize representation of the variable's (matrix or vector)
    size

    - tiling_type -- TilingType (row-major, column-major, or block)

    - dist -- List of localities this Vertex is distributed across
    In the case where data is not distributed across a proper subset of
    available localities, (that is, when all available localities are used
    in distributing this data structure) loc_list is simply empty
    """
    # Since we're gonna start using these for tracking available tiling configs
    # in the expressions, these should really be an Enum. But I don't currently
    # have time for that. Je to škoda
    options = ['row', 'col', 'block']

    def __init__(self, var_name, size, generation, tiling_type, dist):
        super(Vertex, self).__init__(var_name)
        # Add generation value to track how many times this variable changes
        self.size = size
        # The combination of the var_name and generation should give a
        # globally unique identifier to the data enclosed therein.
        # This is mainly because the generation can't be the same for
        # two different reassignments (it might be if we genuinely parsed
        # conditional structures) due to how the dependency graph calculation
        # works
        self.generation = generation
        assert tiling_type in self.options
        self.dist = dist
        self.idx = self.options.index(self.tiling_type)
        self.start_idx = self.idx

    def next(self, nodes, my_idx=0, presence=None):
        if presence is None:
            presence = set()
        if self.name not in presence:
            next_val = (self.idx + 1) % len(self.options)
            self.idx = next_val
            if next_val == self.start_idx and my_idx+1 < len(nodes):
                presence.add(self.name)
                resp = nodes[my_idx + 1].next(nodes, my_idx + 1, presence)
                return False | resp
            elif next_val == self.start_idx and my_idx+1 == len(nodes):
                return True
        elif my_idx+1 < len(nodes):
            return False | nodes[my_idx + 1].next(nodes, my_idx + 1, presence)
        elif my_idx+1 == len(nodes):
            return True
        else:
            return False
        return False

    def get_opposite_idx(self):
        if self.idx == 0:
            return 1
        elif self.idx == 1:
            return 0
        else:
            return 2

    @property
    def tiling_type(self):
        return self.options[self.idx]

    @property
    def flip_size(self):
        if self.size == MatrixSize.large_small:
            return MatrixSize.small_large
        elif self.size == MatrixSize.small_large:
            return MatrixSize.small_large
        return self.size


    def __str__(self):
        return self.name + " " + self.options[self.idx] + " " + str(self.idx)

    def __repr__(self):
        return self.name + " " + self.options[self.idx] + " " + str(self.idx)
