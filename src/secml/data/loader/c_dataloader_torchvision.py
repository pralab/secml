"""
.. module:: DataLoaderTorchDataset
   :synopsis: Loader for Torchvision datasets

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""
from secml.array import CArray
from secml.data import CDataset
from secml.data.loader import CDataLoader

from secml.settings import SECML_DS_DIR
from secml.utils import fm


class CDataLoaderTorchDataset(CDataLoader):
    """Wrapper for loading Torchvision datasets as CDatasets.

    Parameters
    ----------
    tv_dataset_class : torch.Dataset
        torchvision dataset class to load

    """

    def __init__(self, tv_dataset_class, **kwargs):
        root = kwargs.pop('root', fm.join(SECML_DS_DIR, 'pytorch'))
        self._tv_dataset = tv_dataset_class(root=root, **kwargs)
        self._class_to_idx = self._tv_dataset.class_to_idx

    def load(self, *args, **kwargs):
        patterns, labels = self._tv_dataset.data, self._tv_dataset.targets
        patterns = CArray(patterns.view(len(labels), -1).numpy())
        labels = CArray(labels.numpy())
        return CDataset(patterns, labels)

    @property
    def class_to_idx(self):
        """Dictionary for matching indexes and class names"""
        return self._class_to_idx
