import unittest

from problem import Problem
from solve import greedy_solve, local_solve
from util_test import make_basic_edge_set, generate_random_problem, \
    generate_entire_program, make_three_level_edge_set, \
    make_multi_component_edge_set, make_basic_edge_set_add_transpose, \
    run_four_tests

from detail import EdgeSpace, get_level_sets
from matrix_size import MatrixSize
from ops.add import Add
from ops.inv import Inv
from ops.mul import Mul
from ops.transpose import Transpose
from problem import Problem
from solve import local_solve, greedy_solve


class TestPerformance(unittest.TestCase):

    def edge_set_pca_3_rounds(self):
        edge_set = {Mul(0, 'b_1_int', ['A', 'b_0']),
                    Mul(1, 'b_1_int_norm', ['b_1_int', 'b_1_int']),
                    Mul(2, 'b_1', ['b_1_int', 'b_1_int_norm']),
                    Mul(3, 'b_2_int', ['A', 'b_1']),
                    Mul(4, 'b_2_int_norm', ['b_2_int', 'b_2_int']),
                    Mul(5, 'b_2', ['b_2_int', 'b_2_int_norm']),
                    Mul(6, 'b_3_int', ['A', 'b_2']),
                    Mul(7, 'b_3_int_norm', ['b_3_int', 'b_3_int']),
                    Mul(8, 'b_3', ['b_3_int', 'b_3_int_norm']),
                    Mul(9, 'alpha', ['b_3', 'A']),
                    Mul(10, 'alpha_2', ['b_2', 'alpha']),
                    Inv(11, 'beta', ['alpha_2']),
                    Mul(12, 'kappa', ['beta', 'AT']),
                    Mul(13, 'gamma', ['kappa', 'b_3']),
                    Mul(14, 'delta', ['A', 'gamma']),
                    Add(15, 'epsilon', ['delta', 'b_3'])
                    }
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['A', 'b_0']]
        return edge_set, vertex_sizes

    def edge_set1(self):
        edge_set = {Mul(0, 'alpha', ['aT', 'a']),
                    Inv(1, 'beta', ['alpha']),
                    Mul(2, 'kappa', ['beta', 'aT']),
                    Mul(3, 'gamma', ['kappa', 'b']),
                    Mul(4, 'gamma_p', ['kappa', 'c']),
                    Mul(5, 'delta', ['a', 'gamma']),
                    Mul(6, 'delta_p', ['a', 'gamma_p']),
                    Add(7, 'epsilon', ['delta', 'b']),
                    Add(8, 'epsilon_p', ['delta_p', 'c'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['a', 'b', 'c']]
        return edge_set, vertex_sizes

    def edge_set2(self):
        edge_set = {Add(0, 'aa', ['a', 'bT']),
                    Add(1, 'ab', ['cT', 'd']),
                    Add(2, 'ac', ['e', 'fT']),
                    Add(3, 'ad', ['g', 'hT']),
                    Add(4, 'ae', ['i', 'jT']),
                    Mul(5, 'af', ['aa', 'ab']),
                    Mul(6, 'ag', ['af', 'ac']),
                    Mul(7, 'ah', ['ag', 'ad']),
                    Mul(8, 'ai', ['ah', 'ae']),
                    Inv(9, 'aj', ['ai'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]
        return edge_set, vertex_sizes

    def edge_set3(self):
        edge_set = {Add(0, 'aa', ['a', 'bT']),
                    Mul(1, 'ab', ['cT', 'd']),
                    Add(2, 'ac', ['e', 'fT']),
                    Mul(3, 'ad', ['g', 'hT']),
                    Add(4, 'ae', ['i', 'jT']),
                    Add(5, 'ak', ['aa', 'abT']),
                    Transpose(6, 'al', ['ak']),
                    Mul(7, 'af', ['al', 'ab']),
                    Mul(8, 'ag', ['af', 'ac']),
                    Mul(9, 'ah', ['ag', 'ad']),
                    Mul(10, 'ai', ['ah', 'ae']),
                    Inv(11, 'aj', ['ai'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]
        return edge_set, vertex_sizes

    def edge_set4(self):
        edge_set = {Mul(0, 'out_1', ['A', 'B']),
                    Mul(1, 'out_2', ['out_1', 'out_1']),
                    Mul(2, 'MSE', ['out_2', 'C']),
                    Mul(3, 'MSE_mod', ['out_1', 'C']),
                    Add(4, 'MSE_sum', ['MSE', 'MSE_mod']),
                    Inv(5, 'inv_1', ['MSE_sum'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['A', 'B', 'C']]
        return edge_set, vertex_sizes

    def edge_set5(self):
        edge_set = {Mul(0, 'compare_1', ['A', 'B_1']),
                    Mul(1, 'compare_2', ['A', 'C_1']),
                    Mul(2, 'compare_3', ['B', 'A_1']),
                    Mul(3, 'compare_4', ['B', 'C_1']),
                    Mul(4, 'compare_5', ['C', 'A_1']),
                    Mul(5, 'compare_6', ['C', 'B_1']),
                    Mul(6, 'compare_7', ['A_1', 'B']),
                    Mul(7, 'compare_8', ['A_1', 'C']),
                    Mul(8, 'compare_9', ['B_1', 'A']),
                    Mul(9, 'compare_10', ['B_1', 'C']),
                    Mul(10, 'compare_11', ['C_1', 'A']),
                    Mul(11, 'compare_12', ['C_1', 'B']),
                    Add(12, 'ABSum', ['A', 'B']),
                    Add(13, 'ABCSum', ['ABSum', 'C']),
                    Mul(14, 'compare_p_1', ['ABCSum', 'A_1']),
                    Mul(15, 'compare_p_2', ['ABCSum', 'B_1']),
                    Mul(16, 'compare_p_3', ['ABCSum', 'C_1'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['A', 'B', 'C', 'A_1', 'B_1', 'C_1']]
        return edge_set, vertex_sizes

    def edge_set6(self):
        edge_set = {Mul(0, 'transform_a', ['A', 'A_1']),
                    Mul(1, 'transform_b', ['A', 'B_1']),
                    Mul(2, 'transform_c', ['A', 'C_1']),
                    Add(3, 'sum_int', ['transform_a', 'transform_b']),
                    Add(4, 'sum', ['sum_int', 'transform_c']),
                    Inv(5, 'b_0', ['sum']),
                    Mul(6, 'b_1', ['b_0', 'B']),
                    Mul(7, 'b_2', ['b_1', 'b_0']),
                    Mul(8, 'b_3', ['b_2', 'b_1']),
                    Mul(9, 'b_4', ['b_3', 'b_2']),
                    Mul(10, 'b_5', ['b_4', 'b_3']),
                    Mul(11, 'output_0', ['A_1', 'b_5']),
                    Mul(12, 'output_1', ['B_1', 'b_5']),
                    Mul(13, 'output_2', ['C_1', 'b_5']),
                    Add(14, 'out_sum_int', ['output_0', 'output_1']),
                    Add(15, 'out_sum', ['out_sum_int', 'output_2']),
                    Inv(16, 'result', ['out_sum'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['A', 'B', 'A_1', 'B_1', 'C_1']]
        return edge_set, vertex_sizes

    def edge_set7(self):
        edge_set = {Mul(0, 'q', ['a', 'b']),
                    Add(1, 'k', ['j', 'b']),
                    Add(2, 'l', ['e', 'e']),
                    Transpose(3, 'p', ['d']),
                    Mul(4, 'o', ['g', 'i']),
                    Mul(5, 'r', ['c', 'h']),
                    Transpose(6, 'm', ['c']),
                    Mul(7, 'n', ['j', 'i']),
                    Add(8, 's', ['l', 'i']),
                    Add(9, 't', ['f', 's'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]
        return edge_set, vertex_sizes

    def edge_set8(self):
        edge_set = {Inv(0, 'o', ['i']),
                    Mul(1, 'n', ['h', 'h']),
                    Add(2, 'k', ['c', 'j']),
                    Add(3, 'q', ['c', 'h']),
                    Mul(4, 'l', ['g', 'b']),
                    Add(5, 'm', ['f', 'i']),
                    Mul(6, 's', ['b', 'g']),
                    Add(7, 'r', ['d', 'n']),
                    Mul(8, 't', ['o', 'a']),
                    Mul(9, 'p', ['e', 'o']),
                    Mul(10, 'u', ['m', 'p'])}
        # [s_s, s_l, l_s, l_l]
        vertex_sizes = [[], [], [], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]
        return edge_set, vertex_sizes

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

    def run_basic_problem_add_transpose(self, tau=10, tau_prime=20, b=2, eta=0.1,
                          trivial=False):
        problem = Problem(*make_basic_edge_set_add_transpose())
        return self.run_problem(problem, tau=tau, tau_prime=tau_prime, b=b, eta=eta, trivial=trivial)

    '''
    def test_four_tests_for_Avah(self):
        run_four_tests(*make_basic_edge_set_add_transpose())
    '''

    def test_linear_regression(self):
        run_four_tests(*self.edge_set1())

    def test_pca_three_level(self):
        run_four_tests(*self.edge_set_pca_3_rounds())

    def test_exp2(self):
        run_four_tests(*self.edge_set2())

    def test_exp3(self):
        run_four_tests(*self.edge_set3())

    def test_exp4(self):
        run_four_tests(*self.edge_set4())

    def test_exp5(self):
        run_four_tests(*self.edge_set5())

    def test_exp6(self):
        run_four_tests(*self.edge_set6())

    def test_exp7(self):
        run_four_tests(*self.edge_set7())

    def test_exp8(self):
        run_four_tests(*self.edge_set8())


if __name__ == '__main__':
    unittest.main()
