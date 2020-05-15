import unittest
from detail import get_all_cost_dicts

class TestTilingCalculators(unittest.TestCase):
    def setUp(self):
        pass

    def test_cannon(self):
        self.assertRaises(AssertionError, get_all_cost_dicts()['mul'], ['row', 'row'])


if __name__ == '__main__':
    unittest.main()
