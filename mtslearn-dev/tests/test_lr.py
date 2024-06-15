import unittest
import numpy as np
from mtslearn.lr import train_logistic_regression

class TestLogisticRegression(unittest.TestCase):

    def test_train_logistic_regression(self):
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        model = train_logistic_regression(X, y)
        self.assertEqual(model.coef_.shape[1], 2)
        self.assertEqual(len(model.coef_), 1)

if __name__ == '__main__':
    unittest.main()
