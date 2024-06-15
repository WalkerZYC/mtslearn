import unittest
import numpy as np
from mtslearn.utils import evaluate_model
from mtslearn.logistic_regression import train_logistic_regression

class TestUtils(unittest.TestCase):

    def test_evaluate_model(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[1, 2], [2, 3]])
        y_test = np.array([0, 0])
        model = train_logistic_regression(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        self.assertAlmostEqual(accuracy, 1.0)

if __name__ == '__main__':
    unittest.main()
