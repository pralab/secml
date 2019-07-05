"""
.. module:: CClassifierGradientTestRidge
   :synopsis: Debugging class for the classifier gradient class

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""

from secml.ml.classifiers.gradients.tests.utils.gradient_test_classes import \
    CClassifierGradientTestLinear


class CClassifierGradientTestRidge(CClassifierGradientTestLinear):
    """
    This class implement different functionalities which are useful to test
    the CClassifierGradientRidge class.
    """
    __class_type = 'ridge'
