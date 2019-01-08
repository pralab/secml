"""
.. module:: PytorchOptimizationUtils
   :synopsis: Collection of utilities for torch.optim package

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""

__all__ = ['add_weight_decay']


def add_weight_decay(net, l2_value, skip_list=()):
    """Adds the `weight_decay` parameter only to proper parameters.

    The value of `weight_decay` will not be applied to parameters with
     ndim == 1, the bias and any other parameter inside `skip_list`.

    Parameters
    ----------
    net : torch.model
        PyTorch model.
    l2_value : float
        Value of the `weight_decay` parameter.
    skip_list : tuple
        Tuple with the name of the parameters to skip. If a parameter
         is in this list, `weight_decay` will not be applied to it.

    Returns
    -------
    params : list of dict
        List of dictionaries of parameters with weight_decay applied correctly.
        This follows the per-parameter option of PyTorch.

    Notes
    -----
    https://pytorch.org/docs/stable/optim.html#per-parameter-options
    https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    """
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") \
                or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': l2_value}]
