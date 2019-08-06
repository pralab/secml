"""
.. module:: DataLoaderImages-w-Folder
   :synopsis: Loader of an image dataset where clients are specified as different folders.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.data.loader import CDataLoader
from secml.data import CDataset, CDatasetHeader
from secml.array import CArray
from secml.utils import fm
from secml.utils.dict_utils import load_dict

from PIL import Image
import re


class CDataLoaderImgFolders(CDataLoader):
    """Loads a dataset of images where clients are specified as different folders.

    Attributes
    ----------
    class_type : 'img-folders'

    """
    __class_type = 'img-folders'

    def __init__(self):
        # Does nothing...
        pass

    def load(self, ds_path, img_format,
             label_re=None, label_dtype=None, load_data=True):
        """Load all images of specified format inside given path.

        The following custom CDataset attributes are available:
         - 'id': last `ds_path` folder.
         - 'img_w', 'img_h': size of the images in pixels.
         - 'img_c': images number of channels.
         - Any other custom attribute is retrieved from 'attributes.txt' file.
           Only attributes of `str` type are currently supported.

        Any other custom attribute is retrieved from 'attributes.txt' file.

        Parameters
        ----------
        ds_path : str
            Full path to dataset folder.
        img_format : str
            Format of the files to load.
        label_re : re, optional
            Regular expression that identify the correct label.
            If None, the whole name of the leaf folder will be used as label.
        label_dtype : str or dtype, optional
            Datatype of the labels. If None, labels will be strings.
        load_data : bool, optional
            If True (default) features will be stored.
            Otherwise store the paths to the files with dtype=object.

        """
        # Ensuring 'img_format' always has an extension-like pattern
        img_ext = '.' + img_format.strip('.').lower()

        # Dimensions of each image
        img_w = CArray([], dtype=int)
        img_h = CArray([], dtype=int)
        img_c = CArray([], dtype=int)

        # Each directory inside the provided path will be explored recursively
        # and, if leaf, contained images will be loaded
        patterns, labels, img_w, img_h, img_c = self._explore_dir(
            ds_path, img_w, img_h, img_c, img_ext,
            label_re=label_re, load_data=load_data)

        if label_dtype is not None:  # Converting labels if requested
            labels = labels.astype(label_dtype)

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

        header = CDatasetHeader(id=fm.split(ds_path)[1],
                                img_w=img_w, img_h=img_h, img_c=img_c,
                                **attributes)

        return CDataset(patterns, labels, header=header)

    def _explore_dir(self, dir_path, img_w, img_h, img_c, img_ext,
                     label_re=None, load_data=True):
        """Explore input directory and load files if leaf."""
        # Folders/files will be loaded in alphabetical order
        items_list = sorted(fm.listdir(dir_path))

        # A leaf folder is a folder with only files in it
        leaf = not any(fm.folder_exist(
            fm.join(dir_path, item)) for item in items_list)

        if leaf is True:  # Leaf directory, time to load files!
            return self._load_files(
                dir_path, img_w, img_h, img_c, img_ext,
                label_re=label_re, load_data=load_data)

        # Placeholder for patterns/labels CArray
        patterns = None
        labels = None
        for subdir in items_list:

            subdir_path = fm.join(dir_path, subdir)

            # Only consider folders (there could be also files)
            if not fm.folder_exist(subdir_path):
                continue

            # Explore next subfolder
            patterns_new, labels_new, img_w, img_h, img_c = self._explore_dir(
                subdir_path, img_w, img_h, img_c, img_ext,
                label_re=label_re, load_data=load_data)

            patterns = patterns.append(patterns_new, axis=0) \
                if patterns is not None else patterns_new
            labels = labels.append(labels_new) \
                if labels is not None else labels_new

        return patterns, labels, img_w, img_h, img_c

    def _load_files(self, dir_path, img_w, img_h, img_c, img_ext,
                    label_re=None, load_data=True):
        """Loads any file with given extension inside input folder."""
        # Folders/files will be loaded in alphabetical order
        files_list = sorted(fm.listdir(dir_path))

        # Placeholder for patterns/labels CArray
        patterns = None
        labels = None
        for file_name in files_list:

            # Full path to image file
            file_path = fm.join(dir_path, file_name)

            # Load only files of the specified format
            if fm.splitext(file_name)[1].lower() == img_ext:
                # Opening image in lazy mode (to verify dimensions etc.)
                img = Image.open(file_path)

                # Storing image dimensions...
                img_w = img_w.append(img.width)
                img_h = img_h.append(img.height)
                img_c = img_c.append(len(img.getbands()))

                # If load_data is True, store features, else store path
                if load_data is True:
                    # Storing image as a 2D CArray
                    array_img = CArray(img.getdata()).ravel().atleast_2d()
                else:
                    array_img = CArray([[file_path]])

                # Creating the 2D array patterns x features
                patterns = patterns.append(
                    array_img, axis=0) if patterns is not None else array_img

                # Consider only the directory name to set the label
                dir_name = fm.split(dir_path)[1]
                # label is the image's containing folder name or the re result
                c_id = dir_name if label_re is None \
                    else re.search(label_re, dir_name).group(0)
                labels = labels.append(c_id) if labels is not None \
                    else CArray(c_id)

                self.logger.debug("{:} has been loaded..."
                                  "".format(fm.join(dir_path, file_name)))

        return patterns, labels, img_w, img_h, img_c
