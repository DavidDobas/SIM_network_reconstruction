import numpy as np
import os
import unittest

from src.renormalizable_model import RMEnsemble

class TestRenormalizableModel(unittest.TestCase):
    def test_RMEnsemble(self):
        strengths = np.array([[1,2], [3,4], [5,6]])
        z = 5
        ensemble = RMEnsemble(10, strengths, z, "test_ensemble")
        self.assertIsInstance(ensemble, RMEnsemble) 
        os.remove(os.path.join(ensemble.directory, "test_ensemble.pkl"))


if __name__ == '__main__':
    unittest.main()