"""
.. module:: CSecEvalData
   :synopsis: Security evaluation data for attack classes

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.core import CCreator
from secml.array import CArray
from secml.utils import pickle_utils as pk


class CSecEvalData(CCreator):
    """
    This class is a container for data computed during Classifier Security Evaluation.

    Attributes
    ----------
    class_type : 'standard'

    """
    __super__ = 'CSecEvalData'
    __class_type = 'generic'

    def __init__(self):

        # initialize read-write attribute
        self._param_name = None
        self._param_values = None

        # read-only variables (outputs)
        self._scores = None
        self._Y_pred = None
        self._adv_ds = None
        self._time = None
        self._Y = None
        self._Y_target = None
        self._fobj = None

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    @property
    def param_name(self):
        """Returns the name of the parameter representing
        the attack strenght."""
        return self._param_name

    @param_name.setter
    def param_name(self, value):
        """Sets the name of the parameter representing
        the attack strenght."""
        self._param_name = value

    @property
    def param_values(self):
        """Returns the values of the security-evaluation parameter."""
        return self._param_values

    @param_values.setter
    def param_values(self, value):
        """Sets the values of the security-evaluation parameter."""
        self._param_values = CArray(value)

    @property
    def Y(self):
        """Returns the values of the dataset true labels."""
        return self._Y

    @Y.setter
    def Y(self, value):
        """Sets the values of the dataset true labels."""
        self._Y = value

    @property
    def Y_target(self):
        """Returns the values of the desired predicted labels."""
        return self._Y_target

    @Y_target.setter
    def Y_target(self, value):
        """Sets the values of the desired predicted labels."""
        self._Y_target = value

    @property
    def fobj(self):
        """Return objective function values with the different attack power"""
        return self._fobj

    @fobj.setter
    def fobj(self, value):
        """Sets the values of the objective function computed with different attack power."""
        self._fobj = CArray(value)

    @property
    def scores(self):
        """

        Returns
        -------
        scores: list of CArray
            Contain one element for each attack power value.
            Each element contain score assigned by the classifier to all the
            dataset samples.

        """
        return self._scores

    @scores.setter
    def scores(self, value):
        """Set the score values.

        Paramters
        ---------
        value: list of CArray
            Contain one element for each attack power value.
            Each element contain score assigned by the classifier to all the
            dataset samples.

        """
        self._scores = value

    @property
    def Y_pred(self):
        """
        Returns
        -------
        Y_pred : list of CArray
            Contain one element for each attack power value.
            Each element contain label assigned to all the dataset
            samples from the attack.

        """
        return self._Y_pred

    @Y_pred.setter
    def Y_pred(self, value):
        """

        Parameters
        ----------
        value : list of CArray
            Contain one element for each attack power value.
            Each element contain label assigned to all the dataset
            samples from the attack.

        """
        self._Y_pred = value

    @property
    def adv_ds(self):
        """

        Returns
        -------
        adv_ds : list of CDataset.
            containing one dataset for each different parameter value.

        """
        return self._adv_ds

    @adv_ds.setter
    def adv_ds(self, value):
        """

        Parameters
        ----------
         adv_ds : list of CDataset.
            containing one dataset for each different parameter value.

        """
        self._adv_ds = value

    @property
    def time(self):
        """

        Returns
        -------
        time : CArray (n_patterns, num parameter values)
            Each array row contain the times of the attack for one samples.
            Each row element represent a different attack power.

        """
        return self._time

    @time.setter
    def time(self, value):
        """
        Parameters
        ----------
        time: CArray (n_patterns, num parameter values)
            Each array row contain the times of the attack for one samples.
            Each row element represent a different attack power.

        """
        self._time = CArray(value)

    def save(self, path):
        """Load Security evaluation data from file.

        Save a python dict containing all the results.

        """
        results = {p: getattr(self, p) for p in self.get_params()}
        pk.save(path, results)

    @classmethod
    def load(cls, path):
        """Load Security evaluation data from file.

        Save a python dict containing all the results.

        """
        data = cls()
        data.set_params(pk.load(path))
        return data

