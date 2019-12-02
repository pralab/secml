"""
.. module:: MNISTSVM
   :synopsis: Multiclass SVM to be trained on MNIST dataset

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers import CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA


def mnist_svm():
    ova = CClassifierMulticlassOVA(classifier=CClassifierSVM, C=0.1)
    ova.prepare(10)  # MNIST has 10 classes
    return ova
