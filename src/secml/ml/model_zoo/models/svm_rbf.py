"""
.. module:: SVM
   :synopsis: Linear Support Vector Machine with RBF Kernel

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers import CClassifierSVM
from secml.ml.kernel import CKernelRBF


def svm_rbf():
    """Support Vector Machine with RBF Kernel."""
    return CClassifierSVM(kernel=CKernelRBF())
