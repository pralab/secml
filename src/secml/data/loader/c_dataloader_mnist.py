"""
.. module:: DataLoaderMNIST
   :synopsis: Loader the MNIST Handwritten digits dataset

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import gzip
import struct
from array import array
from multiprocessing import Lock
import numpy as np

from secml.data.loader import CDataLoader
from secml.data import CDataset, CDatasetHeader
from secml.array import CArray
from secml.utils import fm
from secml.utils.download_utils import dl_file_gitlab, md5
from secml.settings import SECML_DS_DIR


MODEL_ZOO_REPO_URL = 'https://gitlab.com/secml/secml-zoo'
MNIST_REPO_PATH = 'datasets/MNIST/'

TRAIN_DATA_FILE = 'train-images-idx3-ubyte.gz'
TRAIN_DATA_MD5 = '6bbc9ace898e44ae57da46a324031adb'
TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte.gz'
TRAIN_LABELS_MD5 = 'a25bea736e30d166cdddb491f175f624'
TEST_DATA_FILE = 't10k-images-idx3-ubyte.gz'
TEST_DATA_MD5 = '2646ac647ad5339dbf082846283269ea'
TEST_LABELS_FILE = 't10k-labels-idx1-ubyte.gz'
TEST_LABELS_MD5 = '27ae3e4e09519cfbb04c329615203637'

MNIST_PATH = fm.join(SECML_DS_DIR, 'mnist')


class CDataLoaderMNIST(CDataLoader):
    """Loads the MNIST Handwritten Digits dataset.

    This dataset has a training set of 60,000 examples,
    and a test set of 10,000 examples. All images are 28 x 28
    black and white 8bit (0 - 255).

    Available at: http://yann.lecun.com/exdb/mnist/

    Attributes
    ----------
    class_type : 'mnist'

    """
    __class_type = 'mnist'
    __lock = Lock()  # Lock to prevent multiple parallel download/extraction

    def __init__(self):

        # Build paths of MNIST dataset
        self.train_data_path = fm.join(MNIST_PATH, 'train-images-idx3-ubyte')
        self.train_labels_path = fm.join(MNIST_PATH, 'train-labels-idx1-ubyte')
        self.test_data_path = fm.join(MNIST_PATH, 't10k-images-idx3-ubyte')
        self.test_labels_path = fm.join(MNIST_PATH, 't10k-labels-idx1-ubyte')

        with CDataLoaderMNIST.__lock:
            # For each file check if already downloaded and extracted
            if not fm.file_exist(self.train_data_path) or \
                    md5(self.train_data_path) != TRAIN_DATA_MD5:
                self._get_data(TRAIN_DATA_FILE, MNIST_PATH,
                               self.train_data_path, TRAIN_DATA_MD5)
            if not fm.file_exist(self.train_labels_path) or \
                    md5(self.train_labels_path) != TRAIN_LABELS_MD5:
                self._get_data(TRAIN_LABELS_FILE, MNIST_PATH,
                               self.train_labels_path, TRAIN_LABELS_MD5)
            if not fm.file_exist(self.test_data_path) or \
                    md5(self.test_data_path) != TEST_DATA_MD5:
                self._get_data(TEST_DATA_FILE, MNIST_PATH,
                               self.test_data_path, TEST_DATA_MD5)
            if not fm.file_exist(self.test_labels_path) or \
                    md5(self.test_labels_path) != TEST_LABELS_MD5:
                self._get_data(TEST_LABELS_FILE, MNIST_PATH,
                               self.test_labels_path, TEST_LABELS_MD5)

    def load(self, ds, digits=tuple(range(0, 10)), num_samples=None):
        """Load all images of specified format inside given path.

        Adapted from: http://cvxopt.org/_downloads/mnist.py

        Extra dataset attributes:
         - 'img_w', 'img_h': size of the images in pixels.
         - 'y_original': array with the original labels (before renumbering)

        Parameters
        ----------
        ds : str
            Identifier of the dataset to download,
            either 'training' or 'testing'.
        digits : tuple
            Tuple with the digits to load. By default all digits are loaded.
        num_samples : int or None, optional
            Number of expected samples in resulting ds.
            If int, an equal number of samples will be taken
            from each class until `num_samples` have been loaded.
            If None, all samples will be loaded.

        """
        if ds == "training":
            data_path = self.train_data_path
            lbl_path = self.train_labels_path
        elif ds == "testing":
            data_path = self.test_data_path
            lbl_path = self.test_labels_path
        else:
            raise ValueError("ds must be 'training' or 'testing'")

        self.logger.info(
            "Loading MNIST {:} dataset from {:}...".format(ds, MNIST_PATH))

        # Opening the labels data
        flbl = open(lbl_path, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        if magic_nr != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic_nr))
        lbl = array("b", flbl.read())
        flbl.close()

        # Opening the images data
        fimg = open(data_path, 'rb')
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        if magic_nr != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic_nr))
        img = array("B", fimg.read())
        fimg.close()

        # Convert digits to tuple in case was passed as array/list
        digits = tuple(digits)

        # Number of samples per class
        if num_samples is not None:
            div = len(digits)
            n_samples_class = [
                int(num_samples / div) + (1 if x < num_samples % div else 0)
                for x in range(div)]
            n_samples_class = {
                e: n_samples_class[e_i] for e_i, e in enumerate(digits)}
        else:  # No constraint on the number of samples
            n_samples_class = {e: size for e in digits}

        # Counter of already taken sample for a class
        count_samples_class = {e: 0 for e in digits}

        # Extract the indices of samples to load
        ind = []
        for k in range(size):
            if lbl[k] in digits:
                # Check the maximum number of samples for current digits
                if count_samples_class[lbl[k]] < n_samples_class[lbl[k]]:
                    ind += [k]
                    count_samples_class[lbl[k]] += 1

        # Number of loaded samples
        num_loaded = sum(count_samples_class.values())

        # Check if dataset has enough samples
        if num_samples is not None and num_loaded < num_samples:
            min_val = min(count_samples_class.values())
            raise ValueError(
                "not enough samples in dataset for one ore more of the "
                "desired classes ({:} available)".format(min_val))

        images = CArray.zeros((len(ind), rows * cols), dtype=np.uint8)
        labels = CArray.zeros(len(ind), dtype=int)
        digs_array = CArray(digits)  # To use find method
        for i in range(len(ind)):
            images[i, :] = CArray(img[
                ind[i] * rows * cols: (ind[i] + 1) * rows * cols])
            labels[i] = CArray(digs_array.find(digs_array == lbl[ind[i]]))

        header = CDatasetHeader(img_w=28, img_h=28, y_original=digits)

        return CDataset(images, labels, header=header)

    def _get_data(self, file_name, dl_folder, output_path, md5sum):
        """Download input datafile, unzip and store in output_path.

        Parameters
        ----------
        file_name : str
            Name of the file to download.
        dl_folder : str
            Path to the folder where to store the downloaded file.
        output_path : str
            Full path of output file.
        md5sum : str
            Expected MD5 of the downloaded file (after unpacking).

        """
        # Download file and unpack
        fh = dl_file_gitlab(
            MODEL_ZOO_REPO_URL, MNIST_REPO_PATH + file_name, dl_folder)
        with gzip.open(fh, 'rb') as infile:
            with open(output_path, 'wb') as outfile:
                for line in infile:
                    outfile.write(line)
        # Remove download zipped file
        fm.remove_file(fh)
        # Check the hash of the downloaded file (unpacked)
        if md5(output_path) != md5sum:
            raise RuntimeError('Something wrong happened while '
                               'downloading the dataset. Please try again.')
