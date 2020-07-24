"""
.. module:: CSecEval
   :synopsis: Security evaluation of classifiers.

.. moduleauthor:: Battista Biggio <battista.biggio@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
import time

from secml.core import CCreator
from secml.array import CArray

from secml.adv.seceval import CSecEvalData
from secml.adv.attacks.c_attack import CAttack


class CSecEval(CCreator):
    """
    This class repeat the security evaluation (where security is measured with
    a given metric) while the power of the attacker increase.

    Parameters
    ----------
    attack : CAttack
        Class that implements an attack (e.g evasion or poisoning)
    param_name : str
        Name of the parameter that represents the increasingly attacker power.
    param_values : CArray
        Array that contains values that `param_name` will assumes during the
        attack (this define how the attacker power increases).
        If the first value is not zero, zero will be added as first value
    save_adv_ds : bool, optional
        If True, the samples at each parameter will be stored. Default False.

    See Also
    --------
    .CAttack : class that implements the attack.
    """

    def __init__(self, attack, param_name, param_values,
                 save_adv_ds=False):

        # initialize read-write attribute
        self._attack = None

        self._save_adv_ds = None

        # set read-write value
        self.attack = attack
        self._save_adv_ds = save_adv_ds

        self._sec_eval_data = CSecEvalData()
        self._sec_eval_data.param_name = param_name
        self._sec_eval_data.param_values = param_values

        if param_name not in self.attack.get_params():
            raise ValueError("param_name ({:}) should be a parameter of the "
                             "attack but it was not found. Run `attack.get_params()` "
                             "for getting the list of available parameters.".format(
                param_name))

        if not self._attack.y_target is None:
            self._sec_eval_data.Y_target = CArray(
                self._attack.y_target).deepcopy()

    ###########################################################################
    #                     READ-WRITE ATTRIBUTES (INPUTS)
    ###########################################################################

    @property
    def attack(self):
        """Return the attack object that is used from CSecEval to perform
        the attack."""
        return self._attack

    @attack.setter
    def attack(self, value):
        """Sets the attack object that will be used from CSecEval to perform
        the attack"""
        self._attack = value

    @property
    def save_adv_ds(self):
        """
        Returns
        -------
        True/False: whether to store or not the manipulated attack sample dataset

        """
        return self._save_adv_ds

    @save_adv_ds.setter
    def save_adv_ds(self, value):
        """
        Set to True/False depending on whether to store or not the
        manipulated attack sample dataset.

        Parameters
        ----------
        value: CBool

        Returns
        -------
        None
        """
        self._save_adv_ds = bool(value)

    ###########################################################################
    #                     READ-ONLY ATTRIBUTES (OUTPUTS)
    ###########################################################################

    @property
    def sec_eval_data(self):
        """
        Get a sec eval data objects.
        It contains the Security Evaluation Results.

        Returns
        -------
        sec_eval_data: CSecEvalData object
                contains classifier security evaluation results
        """
        return self._sec_eval_data

    ###########################################################################
    #                           PUBLIC METHODS
    ###########################################################################

    def run_sec_eval(self, dataset, **kwargs):
        """Performs attack while the power of the attacker (named param_name)
        increase.

        Parameters
        ----------
        dataset : CDataset
            Dataset that contain samples that will be manipulated
            from the attacker while his attack power increase
        kwargs
            Additional keyword arguments for the `CAttack.run` method.

        """
        # store true labels within class
        self._sec_eval_data.Y = CArray(dataset.Y).deepcopy()

        # init predicted labels and scores
        Y_pred = CArray.zeros(shape=(dataset.num_samples,))
        scores = CArray.zeros(shape=(dataset.num_samples, dataset.num_classes))

        # create data structures to store attack output
        self._sec_eval_data.scores = [CArray(scores).deepcopy() for i in range(
            self._sec_eval_data.param_values.size)]
        self._sec_eval_data.Y_pred = [CArray(Y_pred).deepcopy() for i in range(
            self._sec_eval_data.param_values.size)]

        self._sec_eval_data.time = CArray.zeros(
            shape=(self._sec_eval_data.param_values.size,))

        self._sec_eval_data.fobj = CArray.zeros(
            shape=(self._sec_eval_data.param_values.size,))

        # manipulate attack samples
        adv_ds = None
        for k, value in enumerate(self._sec_eval_data.param_values):

            self.logger.info("Attack with " + self._sec_eval_data.param_name +
                             " = " + str(value))

            # Update the value of parameter in attack class
            # (e.g., value of dmax in CEvasion)
            self._attack.set(self._sec_eval_data.param_name, value)

            start_time = time.time()

            # todo change x_init parameter with p_ds_init
            attack_result = tuple(self._attack.run(
                dataset.X, dataset.Y, ds_init=adv_ds, **kwargs))

            # Expanding generic attack results
            y_pred, scores, adv_ds, fobj = attack_result[:4]

            if self.save_adv_ds is True:
                adv_ds = adv_ds.deepcopy() if adv_ds is not None else None
                if self._sec_eval_data.adv_ds is not None:
                    self._sec_eval_data.adv_ds.append(adv_ds)
                else:
                    self._sec_eval_data.adv_ds = [adv_ds]

            self._sec_eval_data.Y_pred[k] = y_pred
            self._sec_eval_data.scores[k] = scores
            self._sec_eval_data.fobj[k] = fobj
            self._sec_eval_data.time[k] = time.time() - start_time

            self.logger.debug("Time: " + str(self._sec_eval_data.time[k]))

    def save_data(self, path):
        """Store Sec Eval data to file."""
        self.sec_eval_data.save(path)

    def load_data(self, path):
        """Restore Sec Eval data from file."""
        self._sec_eval_data = CSecEvalData.load(path)
