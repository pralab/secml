from __future__ import print_function, absolute_import

__all__ = ['accuracy']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # As output and target might have shape [N, 1, C], squeeze them
    output = output.squeeze(1)
    target = target.squeeze(1)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.max(1)[1].expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res