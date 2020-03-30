import unittest
from cost_calculations import mul_cannon_cost


class TestTilingCalculators(unittest.TestCase):
    def setUp(self):
        pass

    def test_cannon(self):
        self.assertRaises(AssertionError, mul_cannon_cost, ['row', 'row'])


if __name__ == '__main__':
    unittest.main()
