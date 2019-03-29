"""
.. module:: CAttackEvasionCleverhans
    :synopsis: Performs one of the Cleverhans Evasion attacks
                against a classifier.

.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
import numpy as np
import tensorflow as tf
from cleverhans.attacks import \
    FastGradientMethod, CarliniWagnerL2, ElasticNetMethod, SPSA, LBFGS, \
    ProjectedGradientDescent, SaliencyMapMethod, MomentumIterativeMethod, \
    MadryEtAl, BasicIterativeMethod, DeepFool

from secml.tf.clvhs.ml.classifiers import CModelCleverhans
from secml.adv.attacks import CAttack
from secml.adv.attacks.evasion import CAttackEvasion
from secml.array import CArray
from secml.core.constants import nan

SUPPORTED_ATTACKS = [
    FastGradientMethod, CarliniWagnerL2, ElasticNetMethod, SPSA, LBFGS,
    ProjectedGradientDescent, SaliencyMapMethod, MomentumIterativeMethod,
    MadryEtAl, BasicIterativeMethod, DeepFool
]


class CAttackEvasionCleverhans(CAttackEvasion):
    """This class is a wrapper of the attacks implemented in the Cleverhans
    library.
    
    Credits: https://github.com/tensorflow/cleverhans.

    Parameters
    ----------
    classifier : CClassifier
        Target classifier on which the efficacy of the computed attack
        points is evaluates
    n_feats : int
        Number of features of the dataset used to train the classifiers.
    surrogate_classifier : CClassifier
        Surrogate classifier against which the attack is computed.
        This is assumed to be already trained on surrogate_data.
    surrogate_data: CDataset
        Used to train the surrogate classifier.
    y_target : int or None, optional
            If None an indiscriminate attack will be performed, else a
            targeted attack to have the samples misclassified as
            belonging to the y_target class.
    clvh_attack_class
        The CleverHans class that implement the attack
    **kwargs
        Any other parameter for the cleverhans attack.

    Notes
    -----
    The current Tensorflow default graph will be used.

    """
    class_type = 'evasion-cleverhans'

    def __init__(self, classifier, surrogate_classifier,
                 n_feats, n_classes, surrogate_data=None, y_target=None,
                 clvh_attack_class=CarliniWagnerL2, **kwargs):

        self._tfsess = tf.Session()

        # store the cleverhans attack parameters
        self._clvrh_params = kwargs

        # Check if the cleverhans attack is supported
        if clvh_attack_class not in SUPPORTED_ATTACKS:
            raise ValueError("This cleverhans attack is not supported yet!")

        self._clvrh_attack_class = clvh_attack_class

        # store the number of features
        self._n_feats = n_feats
        # store the number of dataset classes
        self._n_classes = n_classes

        self._clvrh_clf = None

        CAttackEvasion.__init__(self, classifier=classifier,
                                surrogate_classifier=surrogate_classifier,
                                surrogate_data=surrogate_data,
                                y_target=y_target)

    ###########################################################################
    #                           READ-ONLY ATTRIBUTES
    ###########################################################################

    @property
    def f_eval(self):
        if self._clvrh_clf:
            return self._clvrh_clf.f_eval
        else:
            return 0

    @property
    def grad_eval(self):
        if self._clvrh_clf:
            return self._clvrh_clf.grad_eval
        else:
            return 0

    ###########################################################################
    #                              PRIVATE METHODS
    ###########################################################################

    def _set_solver_classifier(self):
        """This function set the surrogate classifier,
        if differentiable; otherwise, it learns a smooth approximation for
        the nondiff. (surrogate) classifier (e.g., decision tree)
        using an SVM with the RBF kernel."""

        # update the surrogate classifier
        # we skip the function provided by the superclass as we do not need
        # to set xk and we call directly the one of CAttack that instead
        # learn a differentiable classifier
        CAttack._set_solver_classifier(self)

        # create the cleverhans attack object
        self._tfsess.close()
        self._tfsess = tf.Session()

        # wrap the surrogate classifier into a cleverhans classifier
        self._clvrh_clf = CModelCleverhans(
            self._surrogate_classifier, out_dims=self._n_classes)

        # create an istance of the chosen cleverhans attack
        clvrh_attack = self._clvrh_attack_class(
            self._clvrh_clf, sess=self._tfsess)

        # create the placeholder to feed into the attack the initial evasion
        # samples
        self._initial_x_P = tf.placeholder(
            tf.float32, shape=(None, self._n_feats))

        # placeholder used to feed the true or the target label (it is a
        # one-hot encoded vector)
        self._y_P = tf.placeholder(tf.float32, shape=(1, self._n_classes))

        # create the tf operations to generate the attack
        if not self.y_target:
            self._adv_x_T = clvrh_attack.generate(
                self._initial_x_P, y=self._y_P, **self._clvrh_params)
        else:
            self._adv_x_T = clvrh_attack.generate(
                self._initial_x_P, y_target=self._y_P, **self._clvrh_params)

    def _run(self, x0, y0, x_init=None):
        """Perform evasion for a given dmax on a single pattern.

        It solves:
            min_x g(x),
            s.t. c(x,x0) <= dmax

        Parameters
        ----------
        x0 : CArray
            Initial sample.
        y0 : int or CArray
            The true label of x0.
        x_init : CArray or None, optional
            Initialization point. If None, it is set to x0.

        Returns
        -------
        x_opt : CArray
            Evasion sample
        f_opt : float
            Value of objective function on x_opt (from surrogate learner).

        Notes
        -----
        Internally, this class stores the values of
         the objective function and sequence of attack points (if enabled).

        """
        # if data can not be modified by the attacker, exit
        if not self.is_attack_class(y0):
            self._x_seq = x_init
            self._x_opt = x_init
            self._f_opt = nan
            self._f_seq = nan
            return self._x_opt, self._f_opt

        if x_init is None:
            x_init = x0

        if not isinstance(x_init, CArray):
            raise TypeError("Input vectors should be of class CArray")

        self._x0 = x0
        self._y0 = y0
        self._init_solver()

        x = self._x0.atleast_2d().tondarray().astype(np.float32)

        # create a one-hot-encoded vector to feed the true or
        # the y_target label

        one_hot_y = CArray.zeros(shape=(1, self._n_classes),
                                 dtype=np.float32)

        if self.y_target:
            one_hot_y[0, self.y_target] = 1
        else:  # indiscriminate attack
            one_hot_y[0, self._y0.item()] = 1

        self._x_opt = self._tfsess.run(
            self._adv_x_T, feed_dict={self._initial_x_P: x,
                                      self._y_P: one_hot_y.tondarray()})

        return CArray(self._x_opt), nan
