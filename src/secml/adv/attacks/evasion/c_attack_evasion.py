"""
.. module:: CAttackEvasion
   :synopsis: Interface for evasion attacks

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod
import six
from six.moves import range

from secml.adv.attacks import CAttack
from secml.array import CArray
from secml.data import CDataset


@six.add_metaclass(ABCMeta)
class CAttackEvasion(CAttack):
    """Interface for Evasion attacks.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier.
    surrogate_classifier : CClassifier
        Surrogate classifier, assumed to be already trained.
    surrogate_data : CDataset or None, optional
        Dataset on which the the surrogate classifier has been trained on.
        Is only required if the classifier is nonlinear.
    y_target : int or None, optional
            If None an error-generic attack will be performed, else a
            error-specific attack to have the samples misclassified as
            belonging to the `y_target` class.

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

    def run(self, x, y, ds_init=None, *args, **kargs):
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
            x_opt, f_opt = self._run(x[k, :], y[k], x_init=xi, *args, **kargs)

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

    def objective_function(self, x):
        """Objective function.

        Parameters
        ----------
        x : CArray
            Array with points on which the objective function
            should be computed.

        Returns
        -------
        CArray
            Value of the objective function on each point.

        """
        return self._objective_function(x)

    @abstractmethod
    def _run(self, x0, y0, x_init=None):
        """Perform evasion on a single pattern.

        Parameters
        ----------
        x0 : CArray
            Initial sample.
        y0 : int or CArray
            The true label of x0.
        x_init : CArray or None, optional
            Initialization point. If None, it is set to x0.

        """
        raise NotImplementedError
