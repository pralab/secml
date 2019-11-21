"""
.. module:: CAttackEvasionCleverhansLossesMixin
    :synopsis: Mixin class that defines loss functions
        for supported Cleverhans attacks.

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>

"""

import numpy as np

from secml.array import CArray
from secml.ml.classifiers.loss import CLossCrossEntropy


class CAttackEvasionCleverhansLossesMixin(object):
    """Mixin class for defining losses of several supported
    Cleverhans attacks."""
    def _objective_function_cw(self, x):
        l2dist = ((self._x0 - x) ** 2).sum(axis=1).ravel()
        z_labels, z_predicted = self.classifier.predict(
            x, return_decision_function=True)
        y_target = CArray.zeros(shape=(1, self._n_classes),
                                dtype=np.float32)
        # destination point label
        if self.y_target is not None:
            y_target[0, self.y_target] = 1
        else:  # indiscriminate attack
            y_target[0, self._y0] = 1

        z_target = (z_predicted * y_target).sum(axis=1).ravel()
        z_other = ((z_predicted * (1 - y_target) +
                    (z_predicted.min(axis=1) - 1) * y_target)).max(axis=1)
        z_other = z_other.ravel()

        # The following differs from the exact definition given in Carlini
        # and Wagner (2016). There (page 9, left column, last equation),
        # the maximum is taken over Z_other - Z_ target (or Z_target - Z_other
        # respectively) and -confidence. However, it doesn't seem that that
        # would have the desired effect (loss term is <= 0 if and only if
        # the difference between the logit of the target and any other class
        # differs by at least confidence). Hence the rearrangement here.

        c_weight = self._clvrh_attack.initial_const
        self.confidence = self._clvrh_attack.confidence

        if self.y_target is not None:
            # if targeted, optimize for making the target class most likely
            loss = CArray.maximum(z_other - z_target + self.confidence,
                                  CArray.zeros(x.shape[0]))
        else:
            # if untargeted, optimize for making any other class most likely
            loss = CArray.maximum(z_target - z_other + self.confidence,
                                  CArray.zeros(x.shape[0]))
        return c_weight * loss + l2dist

    def _objective_function_cross_entropy(self, x):

        preds, scores = self.classifier.predict(
            x, return_decision_function=True)

        if self.y_target is None:
            target = self._y0
        else:
            target = CArray(self.y_target)
        loss = CLossCrossEntropy()
        f_obj = loss.loss(y_true=target, score=scores)

        return f_obj if self.y_target is not None else -f_obj
