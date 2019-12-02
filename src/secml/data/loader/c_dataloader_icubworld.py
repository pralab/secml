"""
.. module:: CDataLoaderICubWorld28
   :synopsis: Loader of the ICubWorld dataset

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Angelo Sotgiu

"""
from multiprocessing import Lock
import zipfile
import os
from fnmatch import fnmatch

from abc import ABCMeta, abstractmethod

from PIL import Image

from secml import settings
from secml.array import CArray
from secml.data import CDataset, CDatasetHeader
from secml.data.loader import CDataLoader
from secml.data.loader.loader_utils import resize_img, crop_img
from secml.utils import fm
from secml.utils.download_utils import dl_file, md5

# Folder where all iCubWorld dataset will be stored
ICUBWORLD_PATH = fm.join(settings.SECML_DS_DIR, 'iCubWorld')

# iCubWorld28
ICUBWORLD28_URL = \
    'https://data.mendeley.com/datasets/3n2vh9rdxd/1/files/' \
    '9e3a79ef-18d9-4c37-b76c-0c34ead60544/iCubWorld28_128x128.zip?dl=1'
ICUBWORLD28_MD5 = 'd4fcdd02bdb0054688a213611a7a8ae7'
ICUBWORLD28_PATH = fm.join(ICUBWORLD_PATH, 'iCubWorld28')


# TODO: iCubWorld 1.0
# TODO: Hello iCubWorld
# TODO: iCubWorld Transformations


class CDataLoaderICubWorld(CDataLoader, metaclass=ABCMeta):
    """Interface for loaders of iCubWorld datasets.

    iCubWorld is a set of computer vision datasets for robotic applications,
    developed by Istituto Italiano di Tecnologia (IIT), Genova, Italy.

    REF: https://robotology.github.io/iCubWorld

    """

    @abstractmethod
    def load(self, *args, **kwargs):
        """Loads a dataset.

        This method should return a `.CDataset` object.

        """
        raise NotImplementedError


class CDataLoaderICubWorld28(CDataLoaderICubWorld):
    """Loader for iCubWorld28 dataset.

    The dataset consists in 28 objects divided in 7 categories,
    where each category includes 4 objects. For each object there are 4
    different acquisition days for training and 4 for testing, with ~150
    frames per acquisition.

    Attributes
    ----------
    class_type : 'icubworld28'

    """
    __class_type = 'icubworld28'
    __lock = Lock()  # Lock to prevent multiple parallel download/extraction

    def __init__(self):

        self._train_path = fm.join(ICUBWORLD28_PATH, 'train')
        self._test_path = fm.join(ICUBWORLD28_PATH, 'test')

        with CDataLoaderICubWorld28.__lock:
            # Download (if needed) data and extract it
            if not fm.folder_exist(self._train_path) \
                    or not fm.folder_exist(self._test_path):
                self._get_data(ICUBWORLD28_URL, ICUBWORLD28_PATH)

    def load(self, ds_type, day='day4', icub7=False,
             resize_shape=(128, 128), crop_shape=None, normalize=True):
        """Load the dataset.

        The pre-cropped version of the images is loaded, with size 128 x 128.
        An additional resize/crop shape could be passed as input if needed.

        Extra dataset attributes:
          - 'img_w', 'img_h': size of the images in pixels.
          - 'y_orig': CArray with the original labels of the objects.

        Parameters
        ----------
        ds_type : str
            Identifier of the dataset to download, either 'train' or 'test'.
        day : str, optional
            Acquisition day from which to load the images. Default 'day4'.
            The available options are: 'day1', 'day2', 'day3', 'day4'.
        icub7 : bool or int, optional
            If True, load a reduced dataset with 7 objects by
            taking the 3rd object for each category. Default False.
            If int, the Nth object for each category will be loaded.
        resize_shape : tuple, optional
           Images will be resized to (height, width) shape. Default (128, 128).
        crop_shape : tuple or None, optional
            If a tuple, a crop of (height, width) shape will be extracted
            from the center of each image. Default None.
        normalize : bool, optional
            If True, images are normalized between 0-1. Default True.

        Returns
        -------
        CDataset
            Output dataset.

        """
        if ds_type == 'train':
            data_path = self._train_path
        elif ds_type == 'test':
            data_path = self._test_path
        else:
            raise ValueError("use ds_type = {'train', 'test'}.")

        day_path = fm.join(data_path, day)
        if not fm.folder_exist(day_path):
            raise ValueError("{:} not available.".format(day))

        self.logger.info(
            "Loading iCubWorld{:} {:} {:} dataset from {:}".format(
                '7' if icub7 else '28', day, ds_type, day_path))

        icub7 = 3 if icub7 is True else icub7  # Use the 3rd sub-obj by default

        x = None
        y_orig = []
        for obj in sorted(fm.listdir(day_path)):  # Objects (cup, sponge, ..)

            obj_path = fm.join(day_path, obj)

            # Sub-objects (cup1, cup2, ...)
            for sub_obj in sorted(fm.listdir(obj_path)):

                if icub7 and sub_obj[-1] != str(icub7):
                    continue  # Load only the `icub7`th object

                self.logger.debug("Loading images for {:}".format(sub_obj))

                sub_obj_path = fm.join(obj_path, sub_obj)

                for f in sorted(fm.listdir(sub_obj_path)):

                    img = Image.open(fm.join(sub_obj_path, f))

                    if resize_shape is not None:
                        img = resize_img(img, resize_shape)
                    if crop_shape is not None:
                        img = crop_img(img, crop_shape)

                    img = CArray(img.getdata(), dtype='uint8').ravel()
                    x = x.append(img, axis=0) if x is not None else img

                    y_orig.append(sub_obj)  # Label is given by sub-obj name

        # Create the int-based array of labels. Keep original labels in y_orig
        y_orig = CArray(y_orig)
        y = CArray(y_orig).unique(return_inverse=True)[1]

        if normalize is True:
            x /= 255.0

        # Size of images is the crop shape (if any) otherwise, the resize shape
        img_h, img_w = crop_shape if crop_shape is not None else resize_shape

        header = CDatasetHeader(img_w=img_w, img_h=img_h, y_orig=y_orig)

        return CDataset(x, y, header=header)

    def _get_data(self, file_url, dl_folder):
        """Download input datafile, unzip and store in output_path.

        Parameters
        ----------
        file_url : str
            URL of the file to download.
        dl_folder : str
            Path to the folder where to store the downloaded file.

        """
        f_dl = fm.join(dl_folder, 'iCubWorld28_128x128.zip?dl=1')
        if not fm.file_exist(f_dl) or md5(f_dl) != ICUBWORLD28_MD5:
            # Generate the full path to the downloaded file
            f_dl = dl_file(file_url, dl_folder, md5_digest=ICUBWORLD28_MD5)

        self.logger.info("Extracting files...")

        # Extract the content of downloaded file
        zipfile.ZipFile(f_dl, 'r').extractall(dl_folder)
        # Remove downloaded file
        fm.remove_file(f_dl)

        # iCubWorld28 zip file contains a macosx private folder, clean it up
        if fm.folder_exist(fm.join(ICUBWORLD28_PATH, '__MACOSX')):
            fm.remove_folder(fm.join(ICUBWORLD28_PATH, '__MACOSX'), force=True)

        # iCubWorld28 zip file contains a macosx private files, clean it up
        for dirpath, dirnames, filenames in os.walk(ICUBWORLD28_PATH):
            for file in filenames:
                if fnmatch(file, '.DS_Store'):
                    fm.remove_file(fm.join(dirpath, file))

        # Now move all data to an upper folder if needed
        if not fm.folder_exist(self._train_path) \
                or not fm.folder_exist(self._test_path):
            sub_d = fm.join(dl_folder, fm.listdir(dl_folder)[0])
            for e in fm.listdir(sub_d):
                e_full = fm.join(sub_d, e)  # Full path to current element
                try:  # Call copy_file or copy_folder when applicable
                    if fm.file_exist(e_full) is True:
                        fm.copy_file(e_full, dl_folder)
                    elif fm.folder_exist(e_full) is True:
                        fm.copy_folder(e_full, fm.join(dl_folder, e))
                except:
                    pass

            # Check that the main dataset file is now in the correct folder
            if not fm.folder_exist(self._train_path) \
                    or not fm.folder_exist(self._test_path):
                raise RuntimeError("dataset main file not available!")

            # The subdirectory can now be removed
            fm.remove_folder(sub_d, force=True)
