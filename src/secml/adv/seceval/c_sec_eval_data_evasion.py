"""
.. module:: CSecEvalDataEvasion
   :synopsis: Security evaluation data for Evasion attacks

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from secml.adv.seceval import CSecEvalData


class CSecEvalDataEvasion(CSecEvalData):
    """Data computed during Classifier Security Evaluation.

    Attributes
    ----------
    class_type : 'evasion'

    """
    __class_type = 'evasion'

    def __init__(self):
        """Class init."""
        super(CSecEvalDataEvasion, self).__init__()

        self._first_eva = None

    def __clear(self):
        """Reset the object."""
        self._first_eva = None

    def __is_clear(self):
        """Returns True if object is clear."""
        return self._first_eva is None

    ###########################################################################
    #                           READ-WRITE ATTRIBUTES
    ###########################################################################

    @property
    def first_eva(self):
        """A list with the first x_opt that evaded (or None)."""
        return self._first_eva

    @first_eva.setter
    def first_eva(self, value):
        """Sets first_eva."""
        self._first_eva = value
