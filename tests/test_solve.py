import unittest

from problem import Problem
from solve import greedy_solve, local_solve
from util_test import make_basic_edge_set, generate_random_problem, \
    generate_entire_program, make_three_level_edge_set, \
    make_multi_component_edge_set, make_basic_edge_set_add_transpose, \
    run_four_tests


class TestSolver(unittest.TestCase):

    def run_problem(self, problem, tau=10, tau_prime=20,
                    b=2, eta=0.1, trivial=False, verbosity=0, skip_real_exhaustive=False):
        if trivial:
            result = local_solve(problem)
        else:
            result = greedy_solve(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, verbosity=verbosity, skip_real_exhaustive=False)
        print("-----------------------------")
        print("Result: ", result)
        print("-----------------------------")
        return result

    def run_basic_problem(self, tau=10, tau_prime=20, b=2, eta=0.1,
                          trivial=False):
        problem = Problem(*make_basic_edge_set())
        return self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    def run_basic_problem_add_transpose(self, tau=10, tau_prime=20, b=2, eta=0.1,
                          trivial=False):
        problem = Problem(*make_basic_edge_set_add_transpose())
        return self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    def run_three_level_problem(self, tau=10, tau_prime=20, b=2, eta=0.1,
                          trivial=False):
        problem = Problem(*make_three_level_edge_set())
        return self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    def run_two_comp_problem(self, tau=10, tau_prime=20, b=2, eta=0.1, trivial=False):
        problem = Problem(*make_multi_component_edge_set())
        return self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    def run_random_problem(self, seed=None, tau=10, tau_prime=20,
                           b=2, eta=0.1, num_expressions=None, num_input_vars=None, trivial=False, verbosity=0, skip_real_exhaustive=False):
        problem, inputs = generate_random_problem(seed, num_expressions=num_expressions, num_input_vars=num_input_vars)
        result = self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial, verbosity=verbosity, skip_real_exhaustive=False)
        print(generate_entire_program(inputs, problem))
        return result

    '''
    
    def test_trivial_solve(self):
        self.assertEqual(self.run_basic_problem(trivial=True),
                         (5.0,
                          {'a': 'row',
                           'add0': 'normal',
                           'add1': 'normal',
                           'add3': 'normal',
                           'b': 'row',
                           'c': 'col',
                           'd': 'row',
                           'e': 'row',
                           'f': 'row',
                           'g': 'row',
                           'h': 'row',
                           'i': 'row',
                           'mul2': 'dot_d',
                           'mul4': 'dot_d'}))

    def test_trivial_solve_add_transpose(self):
        self.assertEqual(self.run_basic_problem_add_transpose(trivial=True),
                         (5.0,
                          {'a': 'row',
                           'add0': 'normal',
                           'add1': 'normal',
                           'add3': 'normal',
                           'b': 'row',
                           'c': 'col',
                           'd': 'row',
                           'e': 'row',
                           'f': 'row',
                           'g': 'row',
                           'h': 'row',
                           'i': 'row',
                           'mul2': 'dot_d',
                           'mul4': 'dot_d'}))

    def test_solve_implementation_search(self):
        self.run_basic_problem()

    def test_solve_exhaustive(self):
        self.run_basic_problem(tau=90000)

    def test_solve_min_deviance(self):
        # tau is arbitrary right now
        self.run_basic_problem(tau_prime=2)

    def test_trivial_three_solve(self):
        result = self.run_three_level_problem(trivial=True)
        num = result[0]
        self.assertAlmostEqual(num, 3.4)
        self.assertEqual(result[1],
                         {'a': 'block',
                          'add1': 'normal',
                          'b': 'block',
                          'c': 'block',
                          'd': 'block',
                          'e': 'block',
                          'f': 'block',
                          'g': 'block',
                          'mul0': 'cannon',
                          'mul2': 'cannon',
                          'transpose3': 'normal'})

    def test_three_solve_implementation_search(self):
        # Verified by hand, but if the cost tables change
        # so might this solution
        result = self.run_three_level_problem()
        num = result[list(result.keys())[0]][0]
        self.assertAlmostEqual(num, 3.4)
        self.assertEqual(result[list(result.keys())[0]][1],
                         {'a': 'block',
                          'add1': 'normal',
                          'b': 'block',
                          'c': 'block',
                          'd': 'block',
                          'e': 'block',
                          'f': 'block',
                          'g': 'block',
                          'mul0': 'cannon',
                          'mul2': 'cannon',
                          'transpose3': 'normal'})

    def test_three_solve_exhaustive(self):
        result = self.run_three_level_problem(tau=90000)
        num = result[list(result.keys())[0]][0]
        self.assertAlmostEqual(num, 3.4)
        self.assertEqual(result[list(result.keys())[0]][1],
                         {'a': 'block',
                          'add1': 'normal',
                          'b': 'block',
                          'c': 'block',
                          'd': 'block',
                          'e': 'block',
                          'f': 'block',
                          'g': 'block',
                          'mul0': 'cannon',
                          'mul2': 'cannon',
                          'transpose3': 'normal'})

    def test_three_solve_min_deviance(self):
        # Also hand-verified
        result = self.run_three_level_problem(tau_prime=2)
        num = result[list(result.keys())[0]][0]
        self.assertAlmostEqual(num, 4.0)
        self.assertEqual(result[list(result.keys())[0]][1],
                         {'a': 'row',
                          'add1': 'normal',
                          'b': 'row',
                          'c': 'row',
                          'd': 'row',
                          'e': 'row',
                          'f': 'row',
                          'g': 'row',
                          'mul0': 'dot_d',
                          'mul2': 'dot_d',
                          'transpose3': 'normal'})

    def test_two_comp_trivial(self):
        self.run_two_comp_problem(trivial=True)

    def test_two_comp_impl_search(self):
        self.run_two_comp_problem(tau_prime=100)

    def test_two_comp_exhaustive(self):
        self.run_two_comp_problem(tau=600000)

    def test_two_comp_min_deviance(self):
        self.run_two_comp_problem(tau_prime=2)

    def test_bigger_program_min_deviance(self):
        # 28.0, 0.079 sec
        self.run_random_problem(seed=101, tau=2, b=2, eta=0.1)

    def test_bigger_program_implementation_search(self):
        # 16.0, 26 sec
        self.run_random_problem(seed=101, tau_prime=80000, b=2, eta=0.1)

    def test_bigger_program_trivial(self):
        # 86.0, 0.005 sec
        self.run_random_problem(seed=101, trivial=True)

    def test_skinny_program_trivial(self):
        # Solution is 49.0, for 0.006 sec
        self.run_random_problem(seed=101, num_expressions=20,
                                num_input_vars=4, trivial=True)

    def test_skinny_program_exhaustive_search(self):
        # Solution is 38.0, six minutes
        self.run_random_problem(seed=101, num_expressions=8,
                                num_input_vars=4, tau=9000000000000000,
                                tau_prime=80000, b=2, eta=0.1, verbosity=1)

    def test_skinny_program_implementation_search(self):
        # Solution is 38.0, six minutes
        self.run_random_problem(seed=101, num_expressions=20,
                                num_input_vars=4,
                                tau_prime=80000, b=2, eta=0.1)


    def test_skinny_program_min_deviance(self):
        # Solution is 68.0, 1/5th of second
        self.run_random_problem(seed=101, num_expressions=20,
                                num_input_vars=4,
                                tau_prime=2, b=2, eta=0.1)
    '''

    def test_four_tests_for_Avah(self):
        run_four_tests(*make_basic_edge_set_add_transpose())


if __name__ == '__main__':
    unittest.main()
