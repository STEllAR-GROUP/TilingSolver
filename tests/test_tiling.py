import unittest
from detail import get_all_cost_dicts
from ops.mul import Mul

class TestTilingCalculators(unittest.TestCase):
    def setUp(self):
        pass

    def test_cannon(self):
        dicts = get_all_cost_dicts()
        print(dicts)

        def my_func():
            dicts['mul']['cannon'](('row', 'row'))

        self.assertRaises(AssertionError, my_func)


if __name__ == '__main__':
    unittest.main()
