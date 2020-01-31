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
        self.vertex_count = {}
        self.edges = []
        self.vertices = []


class TilingParser(ast.NodeVisitor):
    def __init__(self, state):
        print("hello")
        self.state = state

    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):
        print("h")

    def Name(self, node):
        id, ctx = node.value.id, type(node.value.ctx).__name__
        defined_before = False
        new_definition = (ctx == "Store")
        # See if this has been defined before
        if new_definition:
            for id_list, ctx_list in self.state.vertices:
                if id == id_list and ctx_list == "Store":
                    try:
                        count = self.state.vertex_count[id]
                        new_id = id + "_" + str(count)
                        self.state.vertex_count[id] += 1
                        id = new_id
                    except KeyError:
                        self.state.vertex_count[id] = 1
                        id = id+'_1'



        if (id, ctx) not in self.state.vertices:
            assert(ctx != "Load")
            self.state.vertices.append((id, ctx))
        else:
            if ctx == "Store":
                self.state.vertices.append(id)


class Program:
    def __init__(self, program):
        self.program = program
        self.ast = ast.parse(program)
        # Do a first pass, get all the vertices first
        # Then include edges



