"""
.. module:: DataLoaderLFW
   :synopsis: Loader of the LFW Labeled Faces in the Wild dataset

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from multiprocessing import Lock

from secml.data.loader import CDataLoader
from secml.data import CDataset, CDatasetHeader
from secml.array import CArray
from secml.utils import fm
from secml.settings import SECML_DS_DIR

from sklearn.datasets import fetch_lfw_people


class CDataLoaderLFW(CDataLoader):
    """Loads the LFW Labeled Faces in the Wild dataset.

    This dataset is a collection of JPEG pictures of famous people
    collected on the internet, all details are available on
    the official website:

        http://vis-www.cs.umass.edu/lfw/

    Each picture is centered on a single face.
    Each pixel of each channel (color in RGB) is encoded by a
    float in range 0.0 - 1.0.

    The task is called Face Recognition (or Identification):
        given the picture of a face, find the name of the person given
        a training set (gallery).

    This implementation uses `sklearn.datasets.fetch_lfw_people` module.

    Attributes
    ----------
    class_type : 'lfw'

    """
    __class_type = 'lfw'
    __lock = Lock()  # Lock to prevent multiple parallel download/extraction

    def __init__(self):
        # Does nothing
        pass

    def load(self, min_faces_per_person=None, funneled=True, color=False):
        """Load LFW dataset.

        Extra dataset attributes:
         - 'img_w', 'img_h': size of the images in pixels.
         - 'y_names': tuple with the name string for each class.

        Parameters
        ----------
        min_faces_per_person : int or None, optional
            The extracted dataset will only retain pictures of people
            that have at least min_faces_per_person different pictures.
            Default None, so all db images are returned.
        funneled : bool, optional
            Download and use the images aligned with deep funneling.
            Default True.
        color : bool, optional
            Keep the 3 RGB channels instead of averaging them to a
            single gray level channel. Default False.

        """
        with CDataLoaderLFW.__lock:
            lfw_people = fetch_lfw_people(
                data_home=SECML_DS_DIR, funneled=funneled, resize=1,
                min_faces_per_person=min_faces_per_person, color=color,
                slice_=None, download_if_missing=True)

        x = CArray(lfw_people.data)
        y = CArray(lfw_people.target)

        img_w = lfw_people.images.shape[2]
        img_h = lfw_people.images.shape[1]

        y_names = tuple(lfw_people.target_names.tolist())

        header = CDatasetHeader(img_w=img_w, img_h=img_h, y_names=y_names)

        return CDataset(x, y, header=header)

    @staticmethod
    def clean_tmp():
        """Cleans temporary files created by the DB loader.

        This method deletes the joblib-related files created while loading
        the database.

        Does not delete the downloaded database archive.

        """
        jl_tmp_folder = fm.join(SECML_DS_DIR, 'lfw_home', 'joblib')
        if fm.folder_exist(jl_tmp_folder):
            fm.remove_folder(jl_tmp_folder, force=True)
