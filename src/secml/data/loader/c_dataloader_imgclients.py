"""
.. module:: DataLoaderImages-w-Clients
   :synopsis: Loader of an image dataset where clients are specified in a text file

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from prlib.data.loader import CDataLoader
from prlib.data import CDataset
from prlib.array import CArray
from prlib.utils import fm
from prlib.utils.dict_utils import load_dict

from PIL import Image


class CDataLoaderImgClients(CDataLoader):
    """Loads a dataset of images and corresponding labels from 'clients.txt'."""

    class_type = 'imgclients'

    def __init__(self):
        # Does nothing...
        pass

    def load(self, ds_path, img_format, label_dtype=None, load_data=True):
        """Load all images of specified format inside given path.

        'id' (last `ds_path` folder), 'img_w', 'img_h' will be added
        as custom CDataset attributes.

        Any other custom attribute is retrieved from 'attributes.txt' file.

        Parameters
        ----------
        ds_path : str
            Full path to dataset folder.
        img_format : str
            Format of the files to load.
        label_dtype : str or dtype, optional
            Datatype of the labels. If None, labels will be strings.
        load_data : bool, optional
            If True (default) features will be stored.
            Otherwise store the paths to the files with dtype=object.

        """
        # Labels file MUST be available
        if not fm.file_exist(fm.join(ds_path, 'clients.txt')):
            raise OSError("cannot load clients file.")

        # Ensuring 'img_format' always has an extension-like pattern
        img_ext = '.' + img_format.strip('.').lower()

        # Dimensions of each image
        img_w = CArray([], dtype=int)
        img_h = CArray([], dtype=int)

        # Load files!
        patterns, img_w, img_h = self._load_files(
            ds_path, img_w, img_h, img_ext, load_data=load_data)

        labels = CArray.load(
            fm.join(ds_path, 'clients.txt'), dtype=label_dtype).ravel()

        if patterns.shape[0] != labels.size:
            raise ValueError("patterns ({:}) and labels ({:}) do not have "
                             "the same number of elements.".format(
                                 patterns.shape[0], labels.size))

        # Load the file with extra dataset attributes (optional)
        attributes_path = fm.join(ds_path, 'attributes.txt')
        attributes = load_dict(attributes_path) if \
            fm.file_exist(attributes_path) else dict()

        self.logger.info("Loaded {:} images from {:}...".format(
            patterns.shape[0], ds_path))

        return CDataset(patterns, labels, id=fm.split(ds_path)[1],
                        img_w=img_w, img_h=img_h, **attributes)

    def _load_files(self, ds_path, img_w, img_h, img_ext, load_data=True):
        """Loads any file with given extension inside input folder."""
        # Files will be loaded in alphabetical order
        files_list = sorted(fm.listdir(ds_path))

        # Placeholder for patterns CArray
        patterns = None
        for file_name in files_list:

            # Full path to image file
            file_path = fm.join(ds_path, file_name)

            # Load only files of the specified format
            if fm.splitext(file_name)[1].lower() == img_ext:
                # Opening image in lazy mode (to verify dimensions etc.)
                img = Image.open(file_path)

                # Storing image dimensions...
                img_w = img_w.append(img.size[0])
                img_h = img_h.append(img.size[1])

                # If load_data is True, store features, else store path
                if load_data is True:
                    # Storing image as a 2D Carray
                    array_img = CArray([img.getdata()])
                else:
                    array_img = CArray([[file_path]])

                # Creating the 2D array patterns x features
                patterns = patterns.append(
                    array_img, axis=0) if patterns is not None else array_img

                self.logger.debug("{:} has been loaded..."
                                  "".format(fm.join(ds_path, file_name)))

        return patterns, img_w, img_h
