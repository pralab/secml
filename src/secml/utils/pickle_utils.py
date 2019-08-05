"""
.. module:: PickleWrapper
   :synopsis: Wrapper for cPickle object saving package

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from six.moves import cPickle
import gzip

from secml.utils import fm

# Remember to add any new method to following list
__all__ = ['save', 'load']


def save(file_path, obj):
    """Save object to file using cPickle.

    This functions stores a generic python object into
    a compressed gzip file (`*.gz`).

    Saved objects can be loaded using `.load`.

    Parameters
    ----------
    file_path : str
        Path to destination file.
    obj : object
        Any python object to save.

    Returns
    -------
    obj_path : str
        Full path to the stored object.

    """
    # Adding extension to destination file if user forgot about it...
    file_ext = fm.splitext(file_path)[1]
    file_path = file_path + '.gz' if file_ext != '.gz' else file_path

    # open the reference to target file
    with gzip.open(file_path, 'wb') as f_ref:
        # storing the object with a protocol compatible with python >= 2.3
        # TODO: USE PROTOCOL 3 AFTER TRANSITION TO PYTHON 3
        cPickle.dump(obj, f_ref, protocol=2)

    return fm.join(fm.abspath(file_path), fm.split(file_path)[1])


def load(file_path, encoding='bytes'):
    """Load object from cPickle file.

    Load a generic gzip compressed python object created by `.save`.

    Parameters
    ----------
    file_path : str
        Path to target file to read.
    encoding : str, optional
        Encoding to use for loading the file. Default 'bytes'.

    """
    with gzip.open(file_path, 'rb') as f_ref:
        # Loading and returning the object
        try:  # TODO: REMOVE encoding AFTER TRANSITION TO PYTHON 3
            return cPickle.load(f_ref, encoding=encoding)
        except TypeError:
            return cPickle.load(f_ref)
