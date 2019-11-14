"""
.. module:: DataLoaderCIFAR
   :synopsis: Loader the CIFAR tiny images datasets

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import tarfile
from multiprocessing import Lock
import pickle

from abc import ABCMeta, abstractmethod

import numpy as np

from secml.data.loader import CDataLoader
from secml.data import CDataset, CDatasetHeader
from secml.utils import fm
from secml.utils.download_utils import dl_file, md5
from secml.settings import SECML_DS_DIR


CIFAR10_URL_PYTHON = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL_PYTHON = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'

CIFAR_PATH = fm.join(SECML_DS_DIR, 'cifar')
CIFAR10_PATH = fm.join(CIFAR_PATH, 'cifar-10-batches-py')
CIFAR100_PATH = fm.join(CIFAR_PATH, 'cifar-100-python')


class CDataLoaderCIFAR(CDataLoader, metaclass=ABCMeta):
    """Loads the CIFAR tiny images datasets.

    Available at: https://www.cs.toronto.edu/~kriz/cifar.html

    """
    __lock = Lock()  # Lock to prevent multiple parallel download/extraction

    def __init__(self):

        # Extract the name of the data file from the url
        self.data_file = self.data_url.split('/')[-1]

        # Path to the downloaded dataset file
        data_file_path = fm.join(CIFAR_PATH, self.data_file)

        with CDataLoaderCIFAR.__lock:
            # Download (if needed) data and extract it
            if not fm.file_exist(data_file_path) or \
                    md5(data_file_path) != self.data_md5:
                self._get_data(self.data_url, CIFAR_PATH)
            elif not fm.folder_exist(self.data_path):
                # Downloaded datafile seems valid, extract only
                self._get_data(self.data_url, CIFAR_PATH, extract_only=True)

    @property
    @abstractmethod
    def data_url(self):
        """URL of the datafile. Specific for each dataset type.

        Returns
        -------
        str
            URL of the remote datafile with dataset data.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_md5(self):
        """MD5 digest of the datafile. Specific for each dataset type.

        Returns
        -------
        str
            Expected MD5 digest of the dataset file.

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def data_path(self):
        """URL of the data directory. Specific for each dataset type.

        Returns
        -------
        str
            Path to the folder where dataset data is stored.

        """
        raise NotImplementedError

    @abstractmethod
    def load(self, val_size=0):
        """Load all images of the dataset.

        Each image is flattened. The first 1024 entries contain the red
        channel values, the next 1024 the green, and the final 1024 the blue.
        The image is stored in row-major order, so that the first 32 entries
        of the array are the red channel values of the first row of the image.
        Dtype of images is `uint8`. Dtype of labels is `int32`.

        Extra dataset attributes:
         - 'img_w', 'img_h': size of the images in pixels.
         - 'class_names': dictionary with the original name of each class.

        Parameters
        ----------
        val_size : int, optional
            Size of the validation set.
            Default 0, so no validation  dataset will be returned.

        Returns
        -------
        training_set : CDataset
            Training set.
        test_set : CDataset
            Test set.
        validation_set : CDataset, optional
            Validation set. Returned only if val_size > 0.

        """
        raise NotImplementedError

    def _load(self, train_files, test_files, meta_file,
              labels_key, class_names_key, val_size=0):
        """Load all images of the dataset.

        Adapted from: http://dataset-loading.readthedocs.io/en/latest/_modules/dataset_loading/cifar.html

        Parameters
        ----------
        train_files : list
            List of the files where the training set is stored.
        test_files : list
            List of the files where the test set is stored.
        meta_file : str
            Name of the metafile containing the class names.
        labels_key : bytes
            Dictionary key where the labels are stored.
        class_names_key : bytes
            Dictionary key where the class names are stored.
        val_size : int, optional
            Size of the validation set.
            Default 0, so no validation dataset will be returned.

        Returns
        -------
        training_set : CDataset
            Training set.
        test_set : CDataset
            Test set.
        validation_set : CDataset, optional
            Validation set. Returned only if val_size > 0.

        """
        self.logger.info(
            "Loading {:} dataset from {:}...".format(self.class_type,
                                                     self.data_path))

        def load_files(batches_list):
            # Function that loads the data into memory
            data = None
            labels = None
            for batch in batches_list:
                with open(batch, 'rb') as bf:
                    mydict = pickle.load(bf, encoding='bytes')

                # The labels have different names in the two datasets
                new_data = np.array(mydict[b'data'], dtype='uint8')
                newlabels = np.array(mydict[labels_key], dtype='int32')
                if data is not None:
                    data = np.vstack([data, new_data])
                    labels = np.hstack([labels, newlabels])
                else:
                    data = new_data
                    labels = newlabels

            return data, labels

        # Load training and test sets
        train_data, train_labels = load_files(
            [fm.join(self.data_path, f) for f in train_files])
        test_data, test_labels = load_files(
            [fm.join(self.data_path, f) for f in test_files])

        val_data = None
        val_labels = None
        # Populate the validation set if needed
        if val_size > 0:
            train_data, val_data = np.split(
                train_data, [train_data.shape[0] - val_size])
            train_labels, val_labels = np.split(
                train_labels, [train_labels.shape[0] - val_size])

        # Load the class names from the meta file
        class_names = self._load_class_names(meta_file, class_names_key)

        header = CDatasetHeader(img_w=32, img_h=32, class_names=class_names)

        tr = CDataset(train_data, train_labels, header=header)
        ts = CDataset(test_data, test_labels, header=header)

        # Return training set and test set for sure
        out_datasets = (tr, ts)

        if val_size > 0:
            val = CDataset(val_data, val_labels, header=header)
            # Also return the validation dataset
            out_datasets += (val, )

        return out_datasets

    def _load_class_names(self, meta_file, class_names_key):
        """Load the names for the classes in the CIFAR dataset.

        Parameters
        ----------
        meta_file : str
            Name of the metafile where the labels are stored.
        class_names_key : bytes
            Dictionary key where the labels are stored.

        Returns
        ----------
        dict
            A dictionary with the label of each class.

        """
        meta_file_url = fm.join(self.data_path, meta_file)

        # Load the class-names from the pickled file.
        with open(meta_file_url, 'rb') as mf:
            raw = pickle.load(mf, encoding='bytes')[class_names_key]

        # Convert from binary strings.
        names = {i: x.decode('utf-8') for i, x in enumerate(raw)}

        return names

    def _get_data(self, file_url, dl_folder, extract_only=False):
        """Download input datafile, unzip and store in output_path.

        Parameters
        ----------
        file_url : str
            URL of the file to download.
        dl_folder : str
            Path to the folder where to store the downloaded file.
        extract_only : bool, optional
            If True, only extract data from the datafile. Default False.

        """
        # Generate the full path to the downloaded file
        f = fm.join(dl_folder, self.data_url.split('/')[-1])

        if extract_only is False:
            f_dl = dl_file(file_url, dl_folder, md5_digest=self.data_md5)
            if f != f_dl:
                raise ValueError("Unexpected filename {:}".format(f_dl))

        tarfile.open(name=f, mode='r:gz').extractall(dl_folder)


class CDataLoaderCIFAR10(CDataLoaderCIFAR):
    """Loads the CIFAR-10 tiny images dataset.

    The CIFAR-10 dataset consists of 60000 32x32 colour images in
    10 classes, with 6000 images per class. There are 50000 training
    images and 10000 test images.

    Available at: https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes
    ----------
    class_type : 'CIFAR-10'

    """
    __class_type = 'CIFAR-10'

    @property
    def data_url(self):
        """URL of the remote datafile.

        Returns
        -------
        str
            URL of the remote datafile with dataset data.

        """
        return CIFAR10_URL_PYTHON

    @property
    def data_md5(self):
        """MD5 digest of the datafile.

        Returns
        -------
        str
            Expected MD5 digest of the dataset file.

        """
        return CIFAR10_MD5

    @property
    def data_path(self):
        """URL of the data directory.

        Returns
        -------
        str
            Path to the folder where dataset data is stored.

        """
        return CIFAR10_PATH

    def load(self, val_size=0):
        """Load all images of the dataset."""
        # The CIFAR-10 dataset has 5 different batches for train data
        # and one single batch for test data
        # The metafile is called `batches.meta` and the labels `labels`
        train_files = ['data_batch_' + str(i) for i in range(1, 6)]
        test_files = ['test_batch']
        meta_file = 'batches.meta'
        labels_key = b'labels'
        class_names_key = b'label_names'

        return self._load(train_files, test_files, meta_file,
                          labels_key, class_names_key, val_size)
    load.__doc__ += CDataLoaderCIFAR.load.__doc__


# TODO: MANAGE FINE/COARSE LABELS
class CDataLoaderCIFAR100(CDataLoaderCIFAR):
    """Loads the CIFAR-100 tiny images dataset.

    The CIFAR-100 dataset consists of 60000 32x32 colour images in
    100 classes, containing 600 images each. There are 500 training
    images and 100 testing images per class. The 100 classes in the
    CIFAR-100 are grouped into 20 superclasses. Each image comes with a
    "fine" label (the class to which it belongs) and a "coarse" label
    (the superclass to which it belongs).

    Available at: https://www.cs.toronto.edu/~kriz/cifar.html

    Attributes
    ----------
    class_type : 'CIFAR-100'

    """
    __class_type = 'CIFAR-100'

    @property
    def data_url(self):
        """URL of the remote datafile.

        Returns
        -------
        str
            URL of the remote datafile with dataset data.

        """
        return CIFAR100_URL_PYTHON

    @property
    def data_md5(self):
        """MD5 digest of the datafile.

        Returns
        -------
        str
            Expected MD5 digest of the dataset file.

        """
        return CIFAR100_MD5

    @property
    def data_path(self):
        """URL of the data directory.

        Returns
        -------
        str
            Path to the folder where dataset data is stored.

        """
        return CIFAR100_PATH

    def load(self, val_size=0):
        """Load all images of the dataset."""
        # The CIFAR-100 dataset has a single file for train/test
        # The metafile is called `meta` and the labels `fine_labels`
        train_files = ['train']
        test_files = ['test']
        meta_file = 'meta'
        labels_key = b'fine_labels'
        class_names_key = b'fine_label_names'

        return self._load(train_files, test_files, meta_file,
                          labels_key, class_names_key, val_size)
    load.__doc__ = CDataLoaderCIFAR.load.__doc__
