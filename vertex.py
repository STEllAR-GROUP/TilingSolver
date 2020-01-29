import enum


class Size:
    small = 1
    medium = 2
    large = 3


class Edge:
    def __init__(self, op_name, inputs, expression):
        self.op_name = op_name
        self.inputs = inputs
        self.expression = expression
        self.partial_order = []

    def add_partial_order_edge(self, e):
        self.partial_order.append(e)


class Vertex:
    def __init__(self, size, edge_list):
        self.size = size
        # edge_list should be a sparse m
        self.edge_list = edge_list
        self.tiling = None

    def set_tiling(self, tiling):
        self.tiling = tiling
