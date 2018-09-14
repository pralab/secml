"""
.. module:: SoftMax Loss
   :synopsis: Soft Max Loss Function

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
from prlib.classifiers.loss import CLoss
from prlib.classifiers.clf_utils import extend_binary_labels
from prlib.array import CArray


class CLossSoftMax(CLoss):
    """Soft Max Loss Function.
    """

    class_type = 'softmax'
    loss_type = 'classification'

    def __init__(self, extend_binary_labels=False):
        self._extend_binary_labels = bool(extend_binary_labels)

    def _post_prob(self, y, score, c=None):
        """
        Returns p (y|x) for each sample x, w.r.t class label c
        if c is None, we return p(y|x) w.r.t. to the true label y

        Parameters
        ----------
        y: true labels
        score: CArray of shape (n_samples,) (2-class) or (n_samples, n_classes)
        c: class label w.r.t which return the posterior_probs

        Returns
        -------
        Posterior probability for y given x

        """
        score = CArray(score)
        y = CArray(y)

        post_prob = CArray.zeros(shape=(y.size,))

        if len(score.shape) == 1:
            # two-class problems, with score.shape = (num_samples, )
            # reshape score to score.shape = (num_samples, 2)
            score = -score
            score = score.append(-score, axis=0).T

        # this avoids numerical issues (rescaling score values to (-inf, 0])
        score = score - score.max(axis=1)  # broadcasting...

        score_exp = score.exp()
        # TODO: potential CArray BUG - if you ravel score_exp_sum
        # and replace score_exp_sum[y == class_label, :] with
        # score_exp_sum[y == class_label] the output is different.
        score_exp_sum = CArray(score_exp.sum(axis=1))
        if c is None:
            for class_label in y.unique():
                # TODO: other bug in CArray. when all elements of
                # y == class_label are true, it crashes
                post_prob[y == class_label] = \
                    score_exp[y == class_label, class_label] / \
                    score_exp_sum[y == class_label, :]
        else:
            for class_label in y.unique():
                post_prob[y == class_label] = \
                    score_exp[y == class_label, c] / \
                    score_exp_sum[y == class_label, :]

        return post_prob

    def loss(self, y, score, c=None):
        """Compute Soft Max Loss.

        Parameters
        ----------
        y : Vector-like CArray.
            Containing true samples labels
        score : CArray (n_samples * n_classes)
            Matrix that contain score predicted (each row for each class)
        c : int or None, optional
            Class label w.r.t return the posterior probabilities.

        """
        return -CArray.log(self._post_prob(y, score, c=c)).ravel()

    def dloss(self, y, score, c=None):
        """
        Computes the derivative of the softmax loss w.r.t. the discriminant
        function corresponding to class label c

        Assuming c to be i, it is:
            pi-yi, being yi 1 if c is equal to the true label y, 0 otherwise

        If c is None, derivative is taken always w.r.t the true class
        label y, hence we have always p - 1

        """

        # binary case
        if len(score.shape) == 1:
            c = 1

        if c is None:
            c = y

        p = self._post_prob(y, score, c)
        p[y == c] -= 1.0

        return p
