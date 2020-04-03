import random
import sys
import unittest

from problem import Problem


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
        for i in range(4):
            num_vars_in_size = random.randint(0, len(input_vars))
            if i == 3:
                num_vars_in_size = len(input_vars)
            vars_in_size = []
            for j in range(num_vars_in_size):
                elem = random.randint(0, len(input_vars) - 1)
                print(elem, input_vars)
                vars_in_size.append(input_vars[elem])
                input_vars.pop(elem)
            vertex_sizes[i] += vars_in_size
        return vertex_sizes

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
        all_vars = input_var_names.copy()
        edge_set = {}
        new_var_name = chr(97+num_input_vars)
        for i in range(num_expressions):
            op, arity, exp_template = random.choice(self.exp_set)
            while arity > len(all_vars):
                op, arity, exp_template = random.choice(self.exp_set)
            edge_name = op + "_" + str(exp_idx_tracker[op])
            exp_idx_tracker[op] += 1
            out = new_var_name
            new_var_name = chr(ord(new_var_name)+1)
            inputs = []
            available_var_set = all_vars.copy()
            all_vars.append(out)
            for j in range(arity):
                if len(use_all_inputs) > 0:
                    elem = use_all_inputs.pop(0)
                else:
                    elem = random.randint(0, len(available_var_set)-1)
                print(elem, available_var_set)
                inputs.append(available_var_set.pop(elem))
            exp = exp_template.format(out, *inputs)
            edge_set[edge_name] = (op, [out]+inputs, exp)

        print(edge_set, vertex_sizes, 1)
        return Problem(edge_set, vertex_sizes, 1)

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
