import unittest
import numpy as np
from narwhal import util

class DerivativeTests(unittest.TestCase):

    def test_diffmat_first(self):

        D = util.sparse_diffmat(4, 1, 1, order=2)
        self.assertTrue(np.all(D.todense() ==
                        np.array([[-1.5, 2, -0.5, 0],
                                  [-0.5, 0, 0.5, 0],
                                  [0, -0.5, 0, 0.5],
                                  [0, -0.5, 2, -1.5]])))
        return

    def test_deffmat_first2(self):
        D = util.sparse_diffmat(4, 1, 0.5, order=2)
        self.assertTrue(np.all(D.todense() ==
                        np.array([[-3, 4, -1, 0],
                                  [-1, 0, 1, 0],
                                  [0, -1, 0, 1],
                                  [0, -1, 4, -3]], dtype=np.float64)))
        return

    def test_deffmat_first_vector_spacing_regular(self):
        self.assertRaises(NotImplementedError,
                          util.sparse_diffmat,
                          4, 1, np.ones(4), order=2)
        # D = util.sparse_diffmat(4, 1, np.ones(4), order=2)
        # self.assertEqual(D.todense(),
        #         np.array([[-3, 4, -1, 0],
        #                   [-1, 0, 1, 0],
        #                   [0, -1, 0, 1],
        #                   [0, -1, 4, -3]], dtype=np.float64))
        return

    def test_deffmat_first_vector_spacing_irregular(self):
        self.assertRaises(NotImplementedError,
                          util.sparse_diffmat,
                          4, 1, np.arange(4), order=2)
        return

if __name__ == "__main__":
    unittest.main()
