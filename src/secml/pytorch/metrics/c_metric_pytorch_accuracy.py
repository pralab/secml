"""
.. module:: CMetricPyTorchAccuracy
   :synopsis: PyTorch performance metric: Accuracy

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import torch


class CMetricPyTorchAccuracy(object):
    """Performance evaluation metric: Accuracy.

    Accuracy score is the percentage (inside 0/1 range)
    of correctly predicted labels.

    The metric uses:
     - y_true (true ground labels one-hot encoded)
     - score (estimated target values)

    Warnings
    --------
    This metrics is specific for the PyTorch wrapper and only accepts
     and returns `torch.Tensor` objects.

    Examples
    --------
    >>> from secml.pytorch.metrics import CMetricPyTorchAccuracy
    >>> from torch import Tensor

    >>> peval = CMetricPyTorchAccuracy()
    >>> y_true = Tensor([[0, 0, 1],[0, 0, 1],[0, 1, 0]])
    >>> score = Tensor([[1.2, 0., 3.1],[0.5, 2.0, 1.5],[1.2, 0., 3.1]])

    # Prediction @k=1 are: [[2], [1], [2]]
    # Prediction @k=2 are: [[2,0], [1,2], [2,0]]
    # Prediction @k=3 are: [[2,0,1], [1,2,0], [2,0,1]]
    >>> print peval.performance_score(y_true, score, topk=(1,2,3))
    [tensor(33.3333), tensor(66.6667), tensor(100.)]

    """

    def performance_score(self, y_true, score, topk=(1,)):
        """Computes the accuracy@k for the specified values of k.

        Parameters
        ----------
        y_true : torch.Tensor
            True ground labels (one-hot encoded).
            Tensor of shape [N, 1, C] or [N, C].
        score : torch.Tensor
            Estimated target values. Tensor of shape [N, 1, C] or [N, C].
        topk : tuple, optional
            Each value in topk represent the number of scores to take into
            account while computing accuracy, sorted by maximum value.
            Example: if 2, takes into account the top first 2 values.
            See examples for more details. Default (1, ).

        Returns
        -------
        acc : list of torch.Tensor
            List with one single-value tensor with accuracy for each k in topk.

        """
        if not isinstance(y_true, torch.Tensor) or \
                not isinstance(score, torch.Tensor):
            raise TypeError("inputs must be `torch.Tensor` objects")

        maxk = max(topk)
        batch_size = y_true.size(0)

        # As output and target might have shape [N, 1, C], squeeze them
        output = score.squeeze(1)
        target = y_true.squeeze(1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.max(1)[1].expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
