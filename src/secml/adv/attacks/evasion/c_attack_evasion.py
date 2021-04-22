"""
.. module:: CAttackEvasion
   :synopsis: Interface for evasion attacks

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.adv.attacks import CAttack
from secml.core.type_utils import is_int

from secml.array import CArray
from secml.data import CDataset


class CAttackEvasion(CAttack, metaclass=ABCMeta):
    """Interface class for evasion and poisoning attacks.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier (trained).
    y_target : int or None, optional
        If None an error-generic attack will be performed, else a
        error-specific attack to have the samples misclassified as
        belonging to the `y_target` class.
    attack_classes : 'all' or CArray, optional
        Array with the classes that can be manipulated by the attacker or
        'all' (default) if all classes can be manipulated.

    """
    __super__ = 'CAttackEvasion'

    def __init__(self, classifier,
                 y_target=None,
                 attack_classes='all'):

        super(CAttackEvasion, self).__init__(classifier)

        # classes that can be manipulated by the attacker
        self.attack_classes = attack_classes
        self.y_target = y_target

    @property
    def y_target(self):
        return self._y_target

    @y_target.setter
    def y_target(self, value):
        self._y_target = value

    @property
    def attack_classes(self):
        return self._attack_classes

    @attack_classes.setter
    def attack_classes(self, values):
        if not (values == 'all' or isinstance(values, CArray)):
            raise ValueError("`attack_classes` can be 'all' or a CArray")
        self._attack_classes = values

    def is_attack_class(self, y):
        """Returns True/False if the input class can be attacked.

        Parameters
        ----------
        y : int or CArray
            CArray or single label of the class to to be checked.

        Returns
        -------
        bool or CArray
            True if class y can be manipulated by the attacker,
             False otherwise. If CArray, a True/False value for each
             input label will be returned.

        """
        if is_int(y):
            if self._attack_classes == 'all':
                return True  # all classes can be manipulated
            elif CArray(y == self._attack_classes).any():
                return True  # y can be manipulated
            else:
                return False
        elif isinstance(y, CArray):
            v = CArray.zeros(shape=y.shape, dtype=bool)
            if self.attack_classes == 'all':
                v[:] = True  # all classes can be manipulated
                return v
            for i in range(self.attack_classes.size):
                v[y == self.attack_classes[i]] = True  # y can be manipulated
            return v
        else:
            raise TypeError("y can be an integer or a CArray")

    ###########################################################################
    #                                METHODS
    ###########################################################################

    @abstractmethod
    def _run(self, x, y, x_init=None):
        """Optimize the (single) attack point x,y.

        Parameters
        ----------
        x : CArray
            Sample.
        y : int or CArray
            The true label of x.
        x_init : CArray or None, optional
            Initialization point. If None (default), it is set to x.

        Returns
        -------
        x_adv : CArray
                The adversarial example.
        f_opt : float or None, optional
                The value of the objective function at x_adv.

        """
        raise NotImplementedError

    @abstractmethod
    def objective_function(self, x):
        """Objective function.

        Parameters
        ----------
        x : CArray or CDataset

        Returns
        -------
        f_obj : float or CArray of floats

        """
        raise NotImplementedError

    @abstractmethod
    def objective_function_gradient(self, x):
        """Gradient of the objective function."""
        raise NotImplementedError

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
            Mean value of the objective function computed on each data point.

        """
        x = CArray(x).atleast_2d()
        y = CArray(y).atleast_2d()
        x_init = None if ds_init is None else CArray(ds_init.X).atleast_2d()

        # only consider samples that can be manipulated
        v = self.is_attack_class(y)
        idx = CArray(v.find(v)).ravel()

        # number of modifiable samples
        n_mod_samples = idx.size

        adv_ds = CDataset(x.deepcopy(), y.deepcopy())

        # array in which the value of the optimization function are stored
        fs_opt = CArray.zeros(n_mod_samples, )

        for i in range(n_mod_samples):
            k = idx[i].item()  # idx of sample that can be modified

            xi = x[k, :] if x_init is None else x_init[k, :]
            x_opt, f_opt = self._run(x[k, :], y[k], x_init=xi)

            self.logger.info(
                "Point: {:}/{:}, f(x):{:}".format(k, x.shape[0], f_opt))

            adv_ds.X[k, :] = x_opt
            fs_opt[i] = f_opt

        y_pred, scores = self.classifier.predict(
            adv_ds.X, return_decision_function=True)

        y_pred = CArray(y_pred)

        self.logger.info("y_pred after attack:\n{:}".format(y_pred))

        # Return the mean objective function value on the evasion points
        f_obj = fs_opt.mean()

        return y_pred, scores, adv_ds, f_obj
