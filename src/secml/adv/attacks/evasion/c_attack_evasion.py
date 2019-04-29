"""
.. module:: CAttackEvasion
   :synopsis: Class performs evasion attacks against a classifier,
                under different constraints.

.. moduleauthor:: Battista Biggio <battista.biggio@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>
.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
from abc import ABCMeta, abstractmethod
import six
from six.moves import range

from secml.adv.attacks import CAttack
from secml.array import CArray
from secml.data import CDataset


@six.add_metaclass(ABCMeta)
class CAttackEvasion(CAttack):
    """Class that implements evasion attacks.

    It requires classifier, surrogate_classifier, and surrogate_data.
    Note that surrogate_classifier is assumed to be trained (before
    passing it to this class) on surrogate_data.
    
    Parameters
    ----------
    y_target : int or None, optional
            If None an indiscriminate attack will be performed, else a
            targeted attack to have the samples misclassified as
            belonging to the y_target class.

    TODO: complete list of parameters
            
    """
    __super__ = 'CAttackEvasion'

    def __init__(self, classifier,
                 surrogate_classifier,
                 surrogate_data=None,
                 y_target=None):

        super(CAttackEvasion, self).__init__(
            classifier=classifier,
            surrogate_classifier=surrogate_classifier,
            surrogate_data=surrogate_data,
            y_target=y_target)

    ###########################################################################
    #                              PUBLIC METHODS
    ###########################################################################

    def run(self, x, y, ds_init=None):
        """Runs evasion on a dataset.

        Parameters
        ----------
        x : CArray
            Data points.
        y : CArray
            True labels.
        ds_init : CDataset
            Dataset for warm starts.

        Returns
        -------
        y_pred : CArray
            Predicted labels for all ds samples by target classifier.
        scores : CArray
            Scores for all ds samples by target classifier.
        adv_ds : CDataset
            Dataset of manipulated samples.
        f_obj : float
            Average value of the objective function computed on each data point.

        """
        x = CArray(x).atleast_2d()
        y = CArray(y).atleast_2d()
        x_init = None if ds_init is None else CArray(ds_init.X).atleast_2d()

        # only consider samples that can be manipulated
        v = self.is_attack_class(y)
        idx = CArray(v.find(v)).ravel()
        # print(v, idx)

        # number of modifiable samples
        n_mod_samples = idx.size

        adv_ds = CDataset(x.deepcopy(), y.deepcopy())

        # If dataset is sparse, set the proper attribute
        if x.issparse is True:
            self._issparse = True

        # array in which the value of the optimization function are stored
        fs_opt = CArray.zeros(n_mod_samples, )

        for i in range(n_mod_samples):
            k = idx[i].item()  # idx of sample that can be modified

            xi = x[k, :] if x_init is None else x_init[k, :]
            x_opt, f_opt = self._run(x[k, :], y[k], x_init=xi)

            self.logger.info(
                "Point: {:}/{:}, dmax:{:}, f(x):{:}, eval:{:}/{:}".format(
                    k, x.shape[0], self._dmax, f_opt,
                    self.f_eval, self.grad_eval))
            adv_ds.X[k, :] = x_opt
            fs_opt[i] = f_opt

        y_pred, scores = self.classifier.predict(
            adv_ds.X, return_decision_function=True)

        y_pred = CArray(y_pred)

        # Return the mean objective function value on the evasion points (
        # computed from the outputs of the surrogate classifier)
        f_obj = fs_opt.mean()

        return y_pred, scores, adv_ds, f_obj

    @abstractmethod
    def _run(self, x0, y0, x_init=None):
        """
        Move one single point for improve attacker objective function score
        :param x:
        :param y:
        :param x_init:
        :return:
        """
        raise NotImplementedError
