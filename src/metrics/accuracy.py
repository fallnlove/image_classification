from torch import Tensor


class Accuracy:
    def __init__(self):
        self.name = "Accuracy"

    def __call__(self, predictions: Tensor, labels: Tensor, **batch):
        """
        Input:
            predictions (Tensor): predicted probabilities.
            labels (Tensor): labels.
        Output:
            output (int): accuracy metric.
        """

        return (predictions.argmax(-1) == labels).float().sum() / labels.shape[0]
