"""
.. module:: CDataLoaderSvmLight
   :synopsis: Load and save a dataset to/from disk.

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

from secml.data.loader import CDataLoader
from secml.array import CArray
from secml.data import CDataset, CDatasetHeader


class CDataLoaderSvmLight(CDataLoader):
    """Loads and Saves data in svmlight / libsvm format.

    Attributes
    ----------
    class_type : 'svmlight'

    """
    __class_type = 'svmlight'

    def __init__(self):
        # Does nothing...
        pass
    
    def load(self, file_path, dtype_samples=float, dtype_labels=float,
             n_features=None, zero_based=True, remove_all_zero=False,
             multilabel=False, load_infos=False):
        """Loads a dataset from the svmlight / libsvm format and
        returns a sparse dataset.

        Datasets must have only numerical feature indices and
        for every pattern indices must be ordered.

        Extra dataset attributes:
         - 'infos', CArray with inline comment for each sample.

        Parameters
        ----------
        file_path : String
            Path to file were dataset are stored into format svmlight or libsvm.
        dtype_samples : str or dtype, optional
            Data-type to which the samples should be casted. Default is float.
        dtype_labels : str or dtype, optional
            Data-type to which the labels should be casted. Default is float.
        n_features : None or int, optional
            The number of features to use.
            If None (default), it will be inferred. This argument is useful
            to load several files that are subsets of a bigger sliced
            dataset: each subset might not have examples of every feature,
            hence the inferred shape might vary from one slice to another.
        zero_based: bool, optional
            Whether column indices are zero-based (True, default) or
            one-based (False). If column indices are set to be one-based,
            they are transformed to zero-based to match
            Python/NumPy conventions.
        remove_all_zero: boolean, optional, default True
            If True every feature which is zero for every pattern
            will be removed from dataset.
        multilabel : boolean, optional
            True if every sample can have more than one label. Default False.
            (see http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)
        load_infos : bool, optional
            If True, inline comments will be loaded from the svmlight file
            and stored in the infos CDataset parameter (as CArray).
            Default False.

        Returns
        -------
        dataset : CDataset
            Dataset object that contain patterns and labels.
            If `remove_all_zero` is set to True, the returned dataset
            will have the new argument `idx_mapping` with the mapping of
            the returned features to the original features's indices.

        Examples
        --------
        >>> from secml.data.loader import CDataLoaderSvmLight
        >>> from secml.array import CArray
        >>> patterns = CArray ([[1,0,2], [4,0,5]])
        >>> labels = CArray ([0, 1])
        >>> CDataLoaderSvmLight().dump(CDataset(patterns,labels), "myfile.libsvm")
        >>> new_dataset = CDataLoaderSvmLight().load("myfile.libsvm", remove_all_zero=True)
        >>> print(new_dataset.X)  # doctest: +NORMALIZE_WHITESPACE
        CArray(  (0, 1)	2.0
          (0, 0)	1.0
          (1, 1)	5.0
          (1, 0)	4.0)
        >>> print(new_dataset.Y)
        CArray([0. 1.])
        >>> print(new_dataset.header.idx_mapping)
        CArray([0 2])

        """
        # Never use zero_based='auto' in order to avoid
        # any ambiguity with the features indices...
        patterns, labels = load_svmlight_file(file_path,
                                              n_features=n_features,
                                              dtype=float,
                                              multilabel=multilabel,
                                              zero_based=zero_based)

        patterns = CArray(patterns, tosparse=True, dtype=dtype_samples)
        labels = CArray(labels, dtype=dtype_labels)

        header = CDatasetHeader()  # Will be populated with extra attributes

        if remove_all_zero is True:
            patterns, idx_mapping = \
                CDataLoaderSvmLight._remove_all_zero_features(patterns)
            # Store reverse mapping as extra ds attribute
            header.idx_mapping = idx_mapping

        if load_infos is True:
            infos = []
            with open(file_path, 'rt') as f:
                for l_idx, l in enumerate(f):
                    i = l.split(' # ')
                    if len(i) > 2:  # Line should have only one split point
                        raise ValueError("Something wrong happened when "
                                         "extracting infos for line {:}"
                                         "".format(l_idx))
                    infos.append(i[1].rstrip() if len(i) == 2 else '')
            header.infos = CArray(infos)

        if len(header.get_params()) == 0:
            header = None  # Header is empty, store None in ds

        return CDataset(patterns, labels, header=header)

    @staticmethod
    def dump(d, f, zero_based=True, comment=None):
        """Dumps a dataset in the svmlight / libsvm file format.

        This format is a text-based format, with one sample per line. 
        It does not store zero valued features hence is suitable for sparse dataset.
        
        The first element of each line can be used to store a target variable to predict.

        Parameters
        ----------
        d : CDataset 
            Contain dataset with patterns and labels that we want store. 
        f : String 
            Path to file were we want store dataset into format svmlight or libsvm.
        zero_based : bool, optional
            Whether column indices should be written zero-based (True, default) or one-based (False).
        comment : string, optional
            Comment to insert at the top of the file.
            This should be either a Unicode string, which will be encoded as UTF-8,
            or an ASCII byte string. If a comment is given, then it will be preceded
            by one that identifies the file as having been dumped by scikit-learn.
            Note that not all tools grok comments in SVMlight files.

        Examples
        --------
        >>> from secml.data.loader import CDataLoaderSvmLight
        >>> from secml.array import CArray
        >>> patterns = CArray([[1,0,2], [4,0,5]])
        >>> labels = CArray([0,1])
        >>> CDataLoaderSvmLight.dump(CDataset(patterns,labels), "myfile.libsvm")

        """
        dump_svmlight_file(d.X.get_data(), d.Y.get_data(), f,
                           zero_based=zero_based, comment=comment)
