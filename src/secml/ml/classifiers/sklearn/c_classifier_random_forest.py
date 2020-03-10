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
    n_estimators : int, optional
        The number of trees in the forest. Default 10.
    criterion : str, optional
        The function to measure the quality of a split. Supported criteria are
        'gini' (default) for the Gini impurity and 'entropy' for the
        information gain.
    max_depth : int or None, optional
        The maximum depth of the tree. If None (default), then nodes are
        expanded until all leaves are pure or until all leaves contain less
        than min_samples_split samples.
    min_samples_split : int or float, optional
        The minimum number of samples required to split an internal node.
        If int, then consider `min_samples_split` as the minimum number.
        If float, then `min_samples_split` is a fraction and
        `ceil(min_samples_split * n_samples)` are the minimum number of samples
        for each split. Default 2.
    random_state : int, RandomState or None, optional
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`. Default None.
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
