"""
.. module:: PyTorchDataset
   :synopsis: A pytorch dataset with an array of patterns and corresponding labels

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import numpy as np
import torch
from torch.utils.data import Dataset

from secml.array import CArray
from secml.data import CDataset


class CTorchDataset(Dataset):
    """CDataset to Torch Dataset wrapper.

    Parameters
    ----------
    data : CDataset or CArray
        Dataset to be wrapped. Can also be a CArray with the samples and in
         this case the labels can be passed using the `labels` parameter.
    labels : None or CArray
        Labels of the dataset. Can be defined if the samples have been
        passed to the `data` parameter. Input must be a flat array of shape
        (num_samples, ) or a 2-D array with shape (num_samples, num_classes).
    transform : torchvision.transforms or None, optional
        Transformation(s) to be applied to each ds sample.

    """

    def __init__(self, data, labels=None, transform=None):
        """Class constructor."""
        if isinstance(data, CDataset):
            if labels is not None:
                raise TypeError("labels must be defined inside the dataset")
            self.samples = data.X.atleast_2d()
            # Labels inside a CDataset are always stored as flat arrays
            self.labels = data.Y if data.Y is not None else None
        else:
            self.samples = data.atleast_2d()
            self.labels = labels  # 1-D, 2-D or None

        self.transform = transform
        if hasattr(data, 'transform'):
            if transform is not None:
                ValueError("input dataset already defines transforms")
            self.transform = data.transform

    def __len__(self):
        """Returns dataset size."""
        return self.samples.shape[0]

    def __getitem__(self, i):
        """Return desired pair (sample, label) from the dataset."""
        sample = np.array(CArray(self.samples[i, :]).tondarray())

        if self.transform is not None:
            sample = self.transform(sample)

        # Ensure we return tensors
        if not isinstance(sample, torch.Tensor):
            sample = torch.from_numpy(sample)

        if self.labels is not None:
            if self.labels.ndim == 1:  # (num_samples, )
                label = torch.tensor(self.labels[i])
            else:  # (num_samples, num_classes)
                label = np.array(CArray(self.labels[i, :]).tondarray())
                if not isinstance(label, torch.Tensor):
                    label = torch.from_numpy(label)
        else:
            label = torch.tensor(-1)  # Tensor with null label

        return sample.float(), label



