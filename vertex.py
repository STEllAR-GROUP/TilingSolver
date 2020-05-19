from nextable import Nextable


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
    tiling_types = ['row', 'col', 'block']

    def __init__(self, var_name, size, generation, tiling_type, dist):
        super(Vertex, self).__init__()
        # Add generation value to track how many times this variable changes
        self.var_name = var_name
        self.size = size
        # The combination of the var_name and generation should give a
        # globally unique identifier to the data enclosed therein.
        # This is mainly because the generation can't be the same for
        # two different reassignments (it might be if we genuinely parsed
        # conditional structures) due to how the dependency graph calculation
        # works
        self.generation = generation
        assert tiling_type in self.tiling_types
        self.tiling_type = tiling_type
        self.dist = dist
        self.tiling_idx = self.tiling_types.index(self.tiling_type)

    def num_iterations(self):
        return len(self.tiling_types)

    def next(self):
        self.tiling_idx = (self.tiling_idx+1) % len(self.tiling_types)
        self.tiling_type = self.tiling_types[self.tiling_idx]
