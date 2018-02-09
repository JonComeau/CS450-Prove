import unittest
from unittest import TestCase

from O4DecisionTree.DTClassifier import calc_entropy, DecisionTree


class OhFour(TestCase):
    def setup(self):
        pass

    def test_entropy_with_small_set(self):
        self.assertAlmostEqual(
            calc_entropy(['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']),
            0.991076059838222,
            7
        )

    def test_entropy_with_large_set(self):
        self.assertAlmostEqual(
            calc_entropy(["a" if count < 12749 else "b" for count in range(12749 + 50)]),
            0.036876946110275,
            7
        )

    def test_entropy_with_three_classes(self):
        self.assertAlmostEqual(
            calc_entropy([("a" if count < 5 else "b") if count < 65 else "c" for count in range(68)]),
            0.63484572782231,
            7
        )


if __name__ == '__main__':
    unittest.main()
