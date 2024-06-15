import unittest
import numpy as np
from mtslearn.preprocessing import normalize

class TestPreprocessing(unittest.TestCase):

    def test_normalize(self):
        data = np.array([1, 2, 3, 4, 5])
        normalized_data = normalize(data)
        self.assertAlmostEqual(np.mean(normalized_data), 0, places=6)
        self.assertAlmostEqual(np.std(normalized_data), 1, places=6)

if __name__ == '__main__':
    unittest.main()
