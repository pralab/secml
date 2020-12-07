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
        if self._stored_vars is not None and \
                'const' in self._stored_vars:
            stored_const = self._stored_vars['const'][0]
            if self._x0.shape[0] == 1:
                # use same const for all points
                c_weight = stored_const.item()
            else:
                # each point has it own optimized const
                c_weight = CArray.empty(shape=(len(stored_const)))
                for i, c in enumerate(stored_const):
                    c_weight[i] = c
        else:
            self.logger.warning('Constant value not stored during run. Using '
                                'initial_const value. For computing the loss '
                                'with the actual value of const set '
                                '`store_var_list=["const"]` in '
                                'CAttackEvasionCleverhans.__init__().')

            c_weight = self._clvrh_attack.initial_const

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
        # respectively) and -confidence. However, it doesn't seem that
        # would have the desired effect (loss term is <= 0 if and only if
        # the difference between the logit of the target and any other class
        # differs by at least confidence). Hence the rearrangement here.

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

    def _objective_function_elastic_net(self, x):

        if self._stored_vars is not None and \
                'const' in self._stored_vars:
            stored_const = self._stored_vars['const'][0]
            if self._x0.shape[0] == 1:
                # use same const for all points
                c_weight = stored_const.item()
            else:
                # each point has it own optimized const
                c_weight = CArray.empty(shape=(len(stored_const)))
                for i, c in enumerate(stored_const):
                    c_weight[i] = c
        else:
            self.logger.warning('Constant value not stored during run. Using '
                             'initial_const value. For computing the loss '
                             'with the actual value of const set '
                             '`store_var_list=["const"]` in '
                             'CAttackEvasionCleverhans.__init__().')

            c_weight = self._clvrh_attack.initial_const

        if self._clvrh_attack.decision_rule == 'L1':
            d = ((self._x0 - x).abs()).sum(axis=1).ravel()
        elif self._clvrh_attack.decision_rule == 'L2':
            d = ((self._x0 - x) ** 2).sum(axis=1).ravel()
        elif self._clvrh_attack.decision_rule == 'END':
            l1dist = ((self._x0 - x).abs()).sum(axis=1).ravel()
            l2dist = ((self._x0 - x) ** 2).sum(axis=1).ravel()
            d = self._clvrh_attack.beta * l1dist + l2dist
        else:
            raise ValueError("The decision rule only supports `EN`, `L1`, `L2`.")

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
        # respectively) and -confidence. However, it doesn't seem that
        # would have the desired effect (loss term is <= 0 if and only if
        # the difference between the logit of the target and any other class
        # differs by at least confidence). Hence the rearrangement here.
        self.confidence = self._clvrh_attack.confidence

        if self.y_target is not None:
            # if targeted, optimize for making the target class most likely
            loss = CArray.maximum(z_other - z_target + self.confidence,
                                  CArray.zeros(x.shape[0]))
        else:
            # if untargeted, optimize for making any other class most likely
            loss = CArray.maximum(z_target - z_other + self.confidence,
                                  CArray.zeros(x.shape[0]))

        return d + loss * c_weight

    def _objective_function_SPSA(self, x):
        label = self.y_target if self.y_target is not None else self._y0
        preds, scores = self.classifier.predict(x, return_decision_function=True)
        label_logits_mask = CArray.zeros(shape=scores.shape)
        label_logits_mask[:, label] = 1
        highest_nonlabel_logits = scores - label_logits_mask * 9999
        highest_nonlabel_logits = highest_nonlabel_logits.max(axis=1).sum(axis=-1)
        label_logits = scores[:, label].sum(axis=-1)
        loss = highest_nonlabel_logits - label_logits

        loss_multiplier = 1 if self.y_target is not None else -1
        return loss_multiplier * loss
