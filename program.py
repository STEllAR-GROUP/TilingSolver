import ast


def add_one(inputs):
    return 3


def cannon_prod(inputs):
    return 10


def dot_2d2d(inputs):
    return 15


ops = {"add": [add_one], "mul": [cannon_prod, dot_2d2d]}


class TilingProblemBuilder:
    def __init__(self):
        self.op_count = {"add": 0, "mul": 0}
        self.edges = []
        self.vertices = []


class TilingParser(ast.NodeVisitor):
    def __init__(self):
        print("hello")

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        print("h")


class Program:
    def __init__(self, program):
        self.program = program
        self.ast = ast.parse(program)
        # Do a first pass, get all the vertices first
        # Then include edges



