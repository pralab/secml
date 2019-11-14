"""
.. module:: PickleWrapper
   :synopsis: Wrapper for cPickle object saving package

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import pickle
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

    Notes
    -----
    Objects are stored using **protocol 4** data stream format.
    For more information see
    https://docs.python.org/3/library/pickle.html#data-stream-format

    """
    # Adding extension to destination file if user forgot about it...
    file_ext = fm.splitext(file_path)[1]
    file_path = file_path + '.gz' if file_ext != '.gz' else file_path

    # open the reference to target file
    with gzip.open(file_path, 'wb') as f_ref:
        pickle.dump(obj, f_ref, protocol=4)

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
        return pickle.load(f_ref, encoding=encoding)
