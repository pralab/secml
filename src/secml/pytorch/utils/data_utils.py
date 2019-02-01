"""
.. module:: PytorchDatasetUtils
   :synopsis: Collection of utilities for PyTorch datasets

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import torch


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset

    Notes
    -----
    Credits to https://github.com/bearpaw/pytorch-classification

    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std
