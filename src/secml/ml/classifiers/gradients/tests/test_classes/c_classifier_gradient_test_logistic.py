"""
.. module:: CClassifierGradientTestLogisticRegression
   :synopsis: Debugging class for mixin classifier gradient logistic.

.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.classifiers.gradients.tests.test_classes import \
    CClassifierGradientTestLinear


class CClassifierGradientTestLogisticRegression(CClassifierGradientTestLinear):
    __class_type = 'logistic'
