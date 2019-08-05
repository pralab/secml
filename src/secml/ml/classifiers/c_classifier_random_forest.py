"""
.. module:: CClassifierRandomForest
   :synopsis: Random Forest classifier

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.classifiers import CClassifierSkLearn

from sklearn.ensemble import RandomForestClassifier


class CClassifierRandomForest(CClassifierSkLearn):
    """Random Forest classifier.

    Parameters
    ----------
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.

    Attributes
    ----------
    class_type : 'random-forest'

    """
    __class_type = 'random-forest'

    def __init__(self, n_estimators=10, criterion='gini',
                 max_depth=None, min_samples_split=2,
                 random_state=None, preprocess=None):

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        CClassifierSkLearn.__init__(self, sklearn_model=rf,
                                    preprocess=preprocess)
