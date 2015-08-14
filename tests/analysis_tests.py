import unittest
import numpy as np

import narwhal
from narwhal.cast import Cast, CTDCast, XBTCast, LADCP
from narwhal.cast import CastCollection

class CollectionAnalysisTests(unittest.TestCase):

    def test_eofs(self):
        pres = np.arange(1, 300)
        casts = [Cast(depth=np.arange(1, 300), theta=np.sin(pres*i*np.pi/300)) 
                 for i in range(3)]
        cc = CastCollection(casts)
        structures, lamb, eofs = narwhal.analysis.eofs(cc, key="theta")

        self.assertAlmostEqual(np.mean(np.abs(structures["theta_eof1"])), 0.634719360307)
        self.assertAlmostEqual(np.mean(np.abs(structures["theta_eof2"])), 0.363733575635)
        self.assertAlmostEqual(np.mean(np.abs(structures["theta_eof3"])), 0.350665870142)
        self.assertTrue(np.allclose(lamb, [87.27018523, 40.37800904, 2.02016724]))
        return

if __name__ == "__main__":
    unittest.main()
