"""
.. module:: CClassifierDecisionTree
   :synopsis: Decision Tree classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from sklearn import tree

from secml.ml.classifiers import CClassifierSkLearn


class CClassifierDecisionTree(CClassifierSkLearn):
    """Decision Tree Classifier.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'dec-tree'

    """
    __class_type = 'dec-tree'

    def __init__(self, criterion='gini', splitter='best',
                 max_depth=None, min_samples_split=2, preprocess=None):

        dt = tree.DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split)

        CClassifierSkLearn.__init__(self, sklearn_model=dt,
                                    preprocess=preprocess)
