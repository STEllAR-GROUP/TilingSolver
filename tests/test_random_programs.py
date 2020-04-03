import itertools
import random
import sys
import unittest


from problem import Problem
from size import get_valid_input_lists

class TestRandomPrograms(unittest.TestCase):
    MY_SEED = None

    def setUp(self):
        # Tuples of op, arity, and print pattern
        self.exp_set = [("add", 2, "{} = {}+{}"),
                        ("mul", 2, "{} = {}*{}"),
                        ("inv", 1, "{} = ({})^-1"),
                        ("transpose", 1, "{} = ({})^T")]

    def generate_input_var_sizes(self, input_var_names):
        vertex_sizes = [[], [], [], []]
        input_vars = input_var_names.copy()
        for i in range(len(input_vars)):
            size = random.randint(0, 3)
            vertex_sizes[size].append(input_vars[i])
        return vertex_sizes

    def find_valid_algorithms_and_inputs(self, vertex_sizes):
        valid_sizes = get_valid_input_lists()
        options = {i: {} for i in valid_sizes}
        for i in valid_sizes:
            for j in valid_sizes[i]:
                size_tuples = valid_sizes[i][j]
                options_tmp = []
                for k in size_tuples:
                    indices = []
                    # i is synonymous with arity here
                    for l in range(i):
                        indices.append(k[l].value-1)
                    size_tuple_options = vertex_sizes[indices[0]]
                    # Stack products
                    for l in range(1, i+1):
                        size_tuple_options = itertools.product(size_tuple_options, vertex_sizes[indices[l]])
                        size_tuple_options = [(*a, b) for a, b in size_tuple_options]
                    options_tmp += size_tuple_options
                if len(options_tmp) > 0:
                    options[i][j] = options_tmp
        options = {i: options[i] for i in options if len(options[i]) > 0}
        return options

    def generate_random_problem(self):
        my_seed = self.MY_SEED
        print(self.MY_SEED)
        if my_seed is None:
            my_seed = random.randint(0, 100)
        random.seed(my_seed)
        print("Seed is {0}".format(my_seed))
        exp_idx_tracker = {i[0]: 0 for i in self.exp_set}
        num_input_vars = random.randint(1, 8)
        print("Num_input_vars: ", num_input_vars)
        # This max(x , y) is to ensure that there is enough
        # space to use all input variables in at least one expression
        num_expressions = max(num_input_vars, random.randint(7, 20))
        input_var_names = list(range(num_input_vars))
        use_all_inputs = input_var_names.copy()
        input_var_names = [chr(97+i) for i in input_var_names]
        vertex_sizes = self.generate_input_var_sizes(input_var_names)
        beginning_vertex_sizes = vertex_sizes.copy()
        all_vars = input_var_names.copy()
        edge_set = {}
        new_var_name = chr(97+num_input_vars)
        for i in range(num_expressions):
            inputs = []
            available_var_set = all_vars.copy()
            valid_algorithms_and_args = self.find_valid_algorithms_and_inputs(vertex_sizes)
            if len(use_all_inputs) > 0:
                new_valid_algs_and_args = {i: {} for i in valid_algorithms_and_args}
                for i in valid_algorithms_and_args:
                    for j in valid_algorithms_and_args[i]:
                        elem = use_all_inputs.pop(0)
                        new_combos = [x for x in valid_algorithms_and_args[i][j] if elem in x]
                        if len(new_combos) > 0:
                            new_valid_algs_and_args[i][j] = new_combos
                valid_algorithms_and_args = {}
                for i in new_valid_algs_and_args:
                    new_dict = {x for x in new_valid_algs_and_args[i] if len(x) > 0}
                    valid_algorithms_and_args[i] = new_dict





            op, arity, exp_template = random.choice(self.exp_set)
            while arity > len(all_vars):
                op, arity, exp_template = random.choice(self.exp_set)

            for j in range(arity):
                if len(use_all_inputs) > 0:
                    elem = use_all_inputs.pop(0)
                else:
                    elem = random.randint(0, len(available_var_set)-1)
                print(elem, available_var_set)
                inputs.append(available_var_set.pop(elem))
            edge_name = op + "_" + str(exp_idx_tracker[op])
            exp_idx_tracker[op] += 1
            out = new_var_name
            all_vars.append(out)
            new_var_name = chr(ord(new_var_name)+1)
            exp = exp_template.format(out, *inputs)
            edge_set[edge_name] = (op, [out]+inputs, exp)

        print(edge_set, vertex_sizes, 1)
        return Problem(edge_set, beginning_vertex_sizes, 1)

    def test_cost(self):
        print(self.MY_SEED)
        self.MY_SEED = 100
        print(self.MY_SEED)
        problem = self.generate_random_problem()
        problem.get_level_sets()


if __name__ == '__main__':
    if len(sys.argv > 1):
        TestRandomPrograms.MY_SEED = sys.argv.pop()
    TestRandomPrograms.MY_SEED = 100
    unittest.main()
