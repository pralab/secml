"""
.. module:: CClassifierGradientTestLogisticRegression
   :synopsis: Debugging class for the classifier gradient class

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""

from secml.ml.classifiers.gradients.tests.utils import \
    CClassifierGradientTestLinear


class CClassifierGradientTestLogisticRegression(CClassifierGradientTestLinear):
    """
    This class implement different functionalities which are useful to test
    the CClassifierGradientLogistic class.
    """
    __class_type = 'logistic'
