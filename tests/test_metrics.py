# coding=utf-8
import unittest
import torch

from torchsample.metrics import CategoricalAccuracy


class TestMetrics(unittest.TestCase):

    def test_categorical_accuracy(self):
        metric = CategoricalAccuracy()
        predicted = torch.eye(10, requires_grad=True)
        expected = torch.Tensor(list(range(10)), dtype=torch.long, requires_grad=True)
        self.assertEqual(metric(predicted, expected), 100.0)

        # Set 1st column to ones
        predicted = torch.zeros(10, 10, requires_grad=True)
        predicted.data[:, 0] = torch.ones(10)
        self.assertEqual(metric(predicted, expected), 55.0)


if __name__ == '__main__':
    unittest.main()