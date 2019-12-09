"""
.. module:: CDataLoaderPyTorch
   :synopsis: PyTorch loader.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
from torch.utils.data import DataLoader

from secml.data.c_dataset_pytorch import CDatasetPyTorch


class CDataLoaderPyTorch:
    # TODO: ADD DOCSTRING

    def __init__(self, data, labels=None, batch_size=4, shuffle=False,
                 transform=None, num_workers=0):

        self._dataset = CDatasetPyTorch(data,
                                        labels=labels,
                                        transform=transform)

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers

    def get_loader(self):
        data_loader = DataLoader(self._dataset,
                                 batch_size=self._batch_size,
                                 shuffle=self._shuffle,
                                 num_workers=self._num_workers)

        return data_loader
