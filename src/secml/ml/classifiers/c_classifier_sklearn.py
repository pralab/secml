from secml.ml.classifiers import CClassifier
from secml.array import CArray


class CClassifierSkLearn(CClassifier):
    __class_type = 'sklearn-clf'

    def __init__(self, sklearn_model, preprocess=None):
        CClassifier.__init__(self, preprocess=preprocess)
        self._sklearn_model = sklearn_model

    def _fit(self, dataset):
        """Fit sklearn model."""
        self._sklearn_model.fit(dataset.X.get_data(), dataset.Y.get_data())

    def decision_function(self, x, y=None):
        """ TODO docstrings"""

        x = x.atleast_2d()  # Ensuring input is 2-D

        # Transform data if preprocess is defined
        x = self._preprocess_data(x)

        if hasattr(self._sklearn_model, "decision_function"):
            scores = self._sklearn_model.decision_function(x.get_data())
            probs = False
        else:
            scores = self._sklearn_model.predict_proba(x.get_data())
            probs = True

        scores = CArray(scores)

        # two-class classifiers outputting only scores for class 1
        if len(scores.shape) == 1:  # duplicate column for class 0
            outputs = CArray.ones(shape=(x.shape[0], self.n_classes))
            outputs[:, 1] = scores.T
            outputs[:, 0] = -scores.T if probs is False else 1 - scores.T
            scores = outputs

        if scores.shape[1] != self.n_classes:  # this happens in one-vs-one
            raise ValueError(
                "Number of columns is not equal to number of classes!")

        scores.atleast_2d()

        if y is not None:
            return scores[:, y].ravel()
        # else
        return scores

    def predict(self, x, return_decision_function=False, n_jobs=1):
        """Perform classification of each pattern in x.

        If a preprocess has been specified,
         input is normalized before classification.

        Parameters
        ----------
        return_decision_function
        x : CArray
            Array with new patterns to classify, 2-Dimensional of shape
            (n_patterns, n_features).
        return_decision_function : bool, optional
            Whether to return the decision_function value along
            with predictions. Default False.
        n_jobs : int, optional
            Number of parallel workers to use for classification.
            Default 1. Cannot be higher than processor's number of cores.

        Returns
        -------
        labels : CArray
            Flat dense array of shape (n_patterns,) with the label assigned
             to each test pattern. The classification label is the label of
             the class associated with the highest score.
        scores : CArray, optional
            Array of shape (n_patterns, n_classes) with classification
             score of each test pattern with respect to each training class.
            Will be returned only if `return_decision_function` is True.

        Warnings
        --------
        This method implements a generic formulation where the
         classification score is computed separately for training class.
         It's convenient to override this when the score can be computed
         for one of the classes only, e.g. for binary classifiers the score
         for the positive/negative class is commonly the negative of the
         score of the other class.

        """
        scores = self.decision_function(x, y=None)

        # The classification label is the label of the class
        # associated with the highest score
        labels = scores.argmax(axis=1).ravel()

        return (labels, scores) if return_decision_function is True else labels
