import itertools
import random
import sys
import unittest

from matrix_size import MatrixSize
from problem import Problem
from size import get_valid_input_lists, get_output_size_calculators

class TestRandomPrograms(unittest.TestCase):
    MY_SEED = None

    def setUp(self):
        # Tuples of op, arity, and print pattern
        self.exp_set = [("add", 2, "{} = {}+{}"),
                        ("mul", 2, "{} = {}*{}"),
                        ("inv", 1, "{} = ({})^-1"),
                        ("transpose", 1, "{} = ({})^T")]

        self.get_pattern = {"add":"{} = {}+{}",
                            "mul":"{} = {}*{}",
                            "inv":"{} = ({})^-1",
                            "transpose":"{} = {}.T"}

    def generate_input_var_sizes(self, input_var_names):
        vertex_sizes = [[], [], [], []]
        input_vars = input_var_names.copy()
        for i in range(len(input_vars)):
            size = random.randint(0, 3)
            vertex_sizes[size].append(input_vars[i])
        return vertex_sizes

    def find_output_size(self, vertex_sizes, op, inputs):
        operands = []
        for i in inputs:
            found = False
            for j in range(1, 5):
                if i in vertex_sizes[j-1]:
                    operands.append((MatrixSize(j), []))
                    found = True
            if not found:
                raise ValueError("Input not in vertex sizes")
        out_size = get_output_size_calculators()[len(inputs)][op](operands)
        return out_size

    def find_valid_algorithms_and_inputs(self, vertex_sizes):
        valid_sizes = get_valid_input_lists()
        options = {i: {} for i in valid_sizes}
        return_ = []
        for i in valid_sizes:
            for j in valid_sizes[i]:
                size_tuples = valid_sizes[i][j]()
                options_tmp = []
                for k in size_tuples:
                    indices = []
                    # i is synonymous with arity here
                    for l in range(i):
                        indices.append(k[l].value-1)
                    size_tuple_options = vertex_sizes[indices[0]]
                    # Stack products
                    for l in range(1, i):
                        size_tuple_options = itertools.product(size_tuple_options, vertex_sizes[indices[l]])
                        size_tuple_options = [(*a, b) for a, b in size_tuple_options]
                    options_tmp += size_tuple_options
                if len(options_tmp) > 0:
                    options[i][j] = options_tmp
                    return_ += [(i, j, tuple(a)) for a in options_tmp]
        options = {i: options[i] for i in options if len(options[i]) > 0}
        return return_

    def generate_random_problem(self):
        my_seed = self.MY_SEED
        if my_seed is None:
            my_seed = random.randint(0, 100)
        random.seed(my_seed)
        print("Seed is {0}".format(my_seed))
        exp_idx_tracker = {i[0]: 0 for i in self.exp_set}
        num_input_vars = random.randint(1, 8)

        # This max(x , y) is to ensure that there is enough
        # space to use all input variables in at least one expression
        num_expressions = max(num_input_vars, random.randint(7, 20))
        input_var_names = list(range(num_input_vars))
        input_var_names = [chr(97+i) for i in input_var_names]
        self.inputs = input_var_names
        use_all_inputs = input_var_names.copy()
        vertex_sizes = self.generate_input_var_sizes(input_var_names)
        beginning_vertex_sizes = vertex_sizes.copy()
        all_vars = input_var_names.copy()
        prev_layer_added = []
        edge_set = {}
        new_var_name = chr(97+num_input_vars)
        while num_expressions > 0:
            expression_layer_size = random.choices(list(range(1, 5)), [0.3, .3, .3, .1])[0]
            expression_layer_size = min(num_expressions, expression_layer_size)
            new_var_names = []
            new_sizes = [[], [], [], []]
            while expression_layer_size > 0:
                inputs = []
                available_var_set = all_vars.copy()
                valid_algorithms_and_args = self.find_valid_algorithms_and_inputs(vertex_sizes)
                if len(use_all_inputs) > 0:
                    elem = random.choice(use_all_inputs)
                    use_all_inputs.remove(elem)
                else:
                    # This comes from the equation:
                    # q*i+2*i*q = 1.0, which is based on the choice for
                    # p = 2*q, so just created matrices are twice as likely
                    # to be used as those which were instantiated previously
                    i = len(all_vars)
                    j = len(prev_layer_added)
                    q = 1 / (2*i+j)
                    p = 2*q
                    var_selection_set = all_vars+prev_layer_added
                    probabilities = [q for k in all_vars]+[p for k in prev_layer_added]
                    elem = random.choices(var_selection_set, probabilities)[0]
                new_valid_algs_and_args = [(i, j, a) for i, j, a in valid_algorithms_and_args if elem in a]
                arity, op, inputs = random.choice(new_valid_algs_and_args)
                pattern = self.get_pattern[op]
                name = new_var_name
                edge_name = op + "_" + str(exp_idx_tracker[op])
                exp_idx_tracker[op] += 1
                new_var_name = chr(ord(new_var_name) + 1)
                new_var_names.append(name)
                inputs = list(inputs)
                exp = pattern.format(name, *inputs)
                edge_set[edge_name] = (op, [name] + inputs, exp)
                expression_layer_size -= 1
                num_expressions -= 1
                out_size = self.find_output_size(vertex_sizes, op, inputs)
                new_sizes[out_size[0].value-1].append(name)
            for k in range(4):
                vertex_sizes[k] += new_sizes[k]
            all_vars += new_var_names
            prev_layer_added = new_var_names

        return Problem(edge_set, beginning_vertex_sizes, 1)


    def generate_entire_program(self, problem):
        inputs = self.inputs
        small_dim = 100
        large_dim = 3000
        dimension_map = {MatrixSize.small_small: (small_dim, small_dim),
                         MatrixSize.small_large: (small_dim, large_dim),
                         MatrixSize.large_small: (large_dim, small_dim),
                         MatrixSize.large_large: (large_dim, large_dim)}
        program = ""
        for i in inputs:
            row_dim, col_dim = dimension_map[problem.vertices[i].size]
            program += (str(i)+" = random("+str(row_dim)+", "+str(col_dim)+")\n")
        for i in problem.get_level_sets()[1:]:
            while len(i) > 0:
                elem = random.choice(i)
                i.remove(elem)
                program += (str(problem.edges[elem])+'\n')
        return program


    def test_cost(self):
        self.MY_SEED = 100
        problem = self.generate_random_problem()
        print(problem.edges)
        print(problem.get_level_sets())
        print(self.generate_entire_program(problem))


if __name__ == '__main__':
    if len(sys.argv > 1):
        TestRandomPrograms.MY_SEED = sys.argv.pop()
    TestRandomPrograms.MY_SEED = 100
    unittest.main()
