"""
.. module:: CSecEvalDataEvasion
   :synopsis: Security evaluation data for Evasion attacks

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from advlib.seceval import CSecEvalData


class CSecEvalDataEvasion(CSecEvalData):
    """Data computed during Classifier Security Evaluation."""
    class_type = 'evasion'

    def __init__(self):
        """Class init."""

        super(CSecEvalDataEvasion, self).__init__()

        # read-only variables (outputs)
        self._first_eva = None

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

    def __clear(self):
        """Clears run-time used class parameters."""
        self._first_eva = None
