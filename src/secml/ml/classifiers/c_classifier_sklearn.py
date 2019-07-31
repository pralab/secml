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

    def _decision_function(self, x, y=None):
        """Implementation of decision function."""

        if hasattr(self._sklearn_model, "decision_function"):
            scores = self._sklearn_model.decision_function(x.get_data())
            probs = False
        elif hasattr(self._sklearn_model, "predict_proba"):
            scores = self._sklearn_model.predict_proba(x.get_data())
            probs = True
        else:
            raise AttributeError(
                "This model has neither decision_function nor predict_proba.")

        scores = CArray(scores)

        # two-class classifiers outputting only scores for class 1
        if len(scores.shape) == 1:  # duplicate column for class 0
            outputs = CArray.zeros(shape=(x.shape[0], self.n_classes))
            outputs[:, 1] = scores.T
            outputs[:, 0] = -scores.T if probs is False else 1 - scores.T
            scores = outputs

        if scores.shape[1] != self.n_classes:  # this happens in one-vs-one
            raise ValueError(
                "Number of columns is not equal to number of classes!")

        scores.atleast_2d()

        if y is not None:
            return scores[:, y].ravel()
        else:
            return scores
