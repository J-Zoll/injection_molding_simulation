import unittest
import util

class Test_binary_aggregate(unittest.TestCase):
    def test_simple_example(self):
        xs = [.1, .2, .3]
        step_size = .1
        target = [
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ]
        result = util.binary_aggregate(step_size, xs)
        self.assertEqual(result, target)
    

    def test_empty_value(self):
        xs = [.1, .3]
        step_size = .1
        target = [
            [True, False],
            [False, False],
            [False, True]
        ]
        result = util.binary_aggregate(step_size, xs)
        self.assertEqual(result, target)


class Test_discretize(unittest.TestCase):
    def test_discretizes_values(self):
        xs = [0.0, .1, .2, .3, .4, .5]
        step_size = .2
        target_disc_xs = [0.0, .2, .2, .4, .4, .6]
        disc_xs = util.discretize(step_size, xs)
        for res, trg in zip(disc_xs, target_disc_xs):
            self.assertAlmostEqual(res, trg)


if __name__ == "__main__":
    unittest.main()
        