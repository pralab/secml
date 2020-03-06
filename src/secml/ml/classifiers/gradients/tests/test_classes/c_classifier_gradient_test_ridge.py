"""
.. module:: CClassifierGradientTestRidge
   :synopsis: Debugging class for mixin classifier gradient ridge.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTestLinear


class CClassifierGradientTestRidge(CClassifierGradientTestLinear):
    __class_type = 'ridge'
