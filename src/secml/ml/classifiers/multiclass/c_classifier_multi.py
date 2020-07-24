"""
.. module:: CClassifierMulticlass
   :synopsis: Interface for multiclass classifiers

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from abc import ABCMeta, abstractmethod

from secml.ml.classifiers import CClassifier
from secml.utils.dict_utils import merge_dicts


class CClassifierMulticlass(CClassifier, metaclass=ABCMeta):
    """Generic interface for Multiclass Classifiers.

    Parameters
    ----------
    classifier : CClassifier.__class__
        Unbound (not initialized) CClassifier subclass.
    preprocess : CPreProcess or str or None, optional
        Features preprocess to be applied to input data.
        Can be a CPreProcess subclass or a string with the type of the
        desired preprocessor. If None, input data is used as is.
    n_jobs : int
        Number of parallel workers to use for training the classifier.
        Default 1. Cannot be higher than processor's number of cores.

    clf_params : kwargs
        Any other construction parameter for the binary classifiers.

    """
    __super__ = 'CClassifierMulticlass'

    def __init__(self, classifier, preprocess=None, n_jobs=1, **clf_params):
        # Calling init of CClassifier
        super(CClassifierMulticlass, self).__init__(
            preprocess=preprocess, n_jobs=n_jobs)
        # Binary classifier to use
        if not issubclass(classifier, CClassifier):
            raise TypeError(
                "Input classifier must be a subclass of CClassifier")
        # List of binary classifiers
        self._binary_classifiers = [classifier(**clf_params)]

    @CClassifier.verbose.setter
    def verbose(self, level):
        """Set verbosity level and propagate to trained classifiers."""
        # Calling superclass setter of verbose property
        CClassifier.verbose.fset(self, level)
        # Propagate verbosity level to trained binary classifiers
        for i in range(self.num_classifiers):
            self._binary_classifiers[i].verbose = level

    @property
    def classifier(self):
        """Returns the class of the binary classifier used."""
        return self._binary_classifiers[0].__class__

    @property
    def num_classifiers(self):
        """Returns the number of instanced binary classifiers.

        Returns 1 until .fit(dataset) or .prepare(num_classes) is called.

        """
        return len(self._binary_classifiers)

    def set(self, param_name, param_value, copy=False):
        """Set a parameter that has a specific name to a specific value.

        Only parameters, i.e. PUBLIC or READ/WRITE attributes, can be set.

        If setting is performed before training, the parameter to set must
        be a known `.classifier` attribute or a known attribute of any
        parameter already set during or after construction.

        If possible, a reference to the parameter to set is assigned.
        Use `copy=True` to always make a deepcopy before set.

        Parameters
        ----------
        param_name : str
            Name of the parameter to set.
        param_value : any
            Value to set for the parameter. Using a tuple, one value
            for each binary classifier can be specified.
        copy : bool
            By default (False) a reference to the parameter to
            assign is set. If True or a reference cannot be
            extracted, a deepcopy of the parameter is done first.

        """
        # Support for recursive setting, e.g. -> kernel.gamma
        sup_param_name = param_name.split('.', 1)[0]

        # Check if we are setting a parameter of the multiclass classifier
        if hasattr(self, sup_param_name):
            # Call standard set on the multiclass clf object
            super(CClassifierMulticlass, self).set(
                param_name, param_value, copy=copy)
            return

        # SET PARAMETERS OF BINARY CLASSIFIERS
        elif param_name in self._binary_classifiers[0].get_params():
            # Tuples can be used to set a different value for each trained clf
            if isinstance(param_value, tuple):
                # Check if enough binary classifiers are available
                if len(param_value) != self.num_classifiers:
                    raise ValueError("{0} binary classifier instances needed."
                                     " Use .prepare(num_classes={0}) first"
                                     "".format(len(param_value)))
                # Update parameter (different value) in each binary classifier
                for clf_idx, clf in enumerate(self._binary_classifiers):
                    clf.set(param_name, param_value[clf_idx], copy=copy)
            else:
                # Update parameter (same value) in each binary classifier
                for clf in self._binary_classifiers:
                    clf.set(param_name, param_value, copy=copy)
            return

        raise ValueError(
            "cannot set unknown parameter '{:}'".format(param_name))

    def get_state(self):
        """Returns the object state dictionary.

        Returns
        -------
        dict
            Dictionary containing the state of the object.
            The state of the attributes of binary classifiers is a tuple,
            with one value for each binary classifier.

        """
        # Reference state for the binary classifiers
        clf_ref_state = {}
        for attr in self._binary_classifiers[0].get_state():
            clf_ref_state[attr] = []

        # Populate with the values from each binary classifier
        for clf_idx, clf in enumerate(self._binary_classifiers):
            clf_state = clf.get_state()
            for attr in clf_ref_state:
                clf_ref_state[attr].append(clf_state[attr])

        # Append the 'binary_classifier.' identifier in order to distinguish
        # these attributes from the attributes of the main classifier
        # Finally convert to tuple in order to prevent future modifications
        for attr in list(clf_ref_state):
            new_key = 'binary_classifiers.' + attr
            clf_ref_state[new_key] = tuple(clf_ref_state[attr])
            del clf_ref_state[attr]

        # State of the main multiclass classifier
        multi_clf_state = super(CClassifierMulticlass, self).get_state()

        return merge_dicts(multi_clf_state, clf_ref_state)

    def set_state(self, state_dict, copy=False):
        """Sets the object state using input dictionary.

        Only readable attributes of the class,
        i.e. PUBLIC or READ/WRITE or READ ONLY, can be set.

        If possible, a reference to the attribute to set is assigned.
        Use `copy=True` to always make a deepcopy before set.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing the state of the object.
            The state of the attributes of binary classifiers must be
            specified as a tuple, with one value for each binary classifier.
        copy : bool, optional
            By default (False) a reference to the attribute to
            assign is set. If True or a reference cannot be
            extracted, a deepcopy of the attribute is done first.

        """
        for param_name in state_dict:

            # Extract the value of the attribute to set
            param_value = state_dict[param_name]

            # Support for recursive setting, e.g. -> kernel.gamma
            sup_param_name = param_name.split('.', 1)[0]

            # Check if we are setting a parameter of the multiclass classifier
            if hasattr(self, sup_param_name):
                # Call standard set on the multiclass clf object
                super(CClassifierMulticlass, self).set_state(
                    {param_name: param_value}, copy=copy)
                continue

            # SET PARAMETERS OF BINARY CLASSIFIERS
            elif param_name.startswith('binary_classifiers.'):

                # Remove the identifier of the binary classifiers's attributes
                sub_param_name = param_name[len('binary_classifiers.'):]

                if sub_param_name in self._binary_classifiers[0].get_state():

                    if not isinstance(param_value, tuple):
                        raise ValueError("state of attribute '{:}' "
                                         "must specified as a tuple"
                                         "".format(param_name))

                    # Check if enough binary classifiers are available
                    if len(param_value) != self.num_classifiers:
                        raise ValueError(
                            "{0} binary classifier instances needed."
                            " Use .prepare(num_classes={0}) first"
                            "".format(len(param_value)))
                    # Update attribute (different value) in each binary clf
                    for clf_idx, clf in enumerate(self._binary_classifiers):
                        clf.set_state(
                            {sub_param_name: param_value[clf_idx]}, copy=copy)

                    continue

            raise AttributeError(
                "cannot set unknown attribute '{:}'".format(param_name))

    def prepare(self, num_classes):
        """Creates num_classes copies of the binary classifier.

        Creates enough deepcopies of the binary classifier until
        `num_classes` binary classifiers are instanced.
        If `num_classes < self.num_classifiers`,
        classifiers in excess are deleted.

        Parameters
        ----------
        num_classes : int
            Number of binary classifiers to instance.

        """
        from copy import deepcopy
        if num_classes < 1:
            raise ValueError("number of classes must be higher than 0")
        clf = self._binary_classifiers[0]  # Use the first clf as base
        # Create new copies until num_classes binary clf are instanced
        while len(self._binary_classifiers) < num_classes:
            self._binary_classifiers.append(deepcopy(clf))
        # Delete binary classifiers in excess
        del self._binary_classifiers[num_classes:]

    def estimate_parameters(self, dataset, parameters, splitter, metric,
                            pick='first', perf_evaluator='xval'):
        """Estimate parameter that give better result respect a chose metric.

        Parameters
        ----------
        dataset : CDataset
            Dataset to be used for evaluating parameters.
        parameters : dict
            Dictionary with each entry as {parameter: list of values to test}.
            Example:
            `{'C': [1, 10, 100], 'gamma': list(10.0 ** CArray.arange(-4, 4))}`
        splitter : CDataSplitter or str
            Object to use for splitting the dataset into train and validation.
            A splitter type can be passed as string, in this case all
            default parameters will be used. For data splitters, num_folds
            is set to 3 by default.
            See CDataSplitter docs for more information.
        metric : CMetric or str
            Object with the metric to use while evaluating the performance.
            A metric type can be passed as string, in this case all
            default parameters will be used.
            See CMetric docs for more information.
        pick : {'first', 'last', 'random'}, optional
            Defines which of the best parameters set pick.
            Usually, 'first' correspond to the smallest parameters while
            'last' correspond to the biggest. The order is consistent
            to the parameters dict passed as input.
        perf_evaluator : CPerfEvaluator or str, optional
            Performance Evaluator to use. Default 'xval'.

        Returns
        -------
        best_parameters : dict
            Dictionary of best parameters found through performance evaluation.

        """
        # Prepare the multiclass classifier before parameter estimation
        self.prepare(dataset.num_classes)
        # Estimate the best parameters and set them to the binary classifiers
        return super(CClassifierMulticlass, self).estimate_parameters(
            dataset=dataset,
            parameters=parameters,
            splitter=splitter,
            metric=metric,
            pick=pick,
            perf_evaluator=perf_evaluator)

    @abstractmethod
    def _fit(self, x, y):
        """Trains the classifier.

        This method should store the list of trained classifiers
        inside self._trained_classifiers attribute.

        `secml.parallel.parfor2` can be used for parallelization.

        Parameters
        ----------
        x : CArray
            Array to be used for training with shape (n_samples, n_features).
        y : CArray
            Array of shape (n_samples,) containing the class labels.

        Returns
        -------
        CClassifierMulticlass
            Trained classifier.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def binarize_dataset(class_idx, dataset):
        """Returns the dataset needed by the class_idx binary classifier.

        Parameters
        ----------
        class_idx : int
            Index of the target class.
        dataset : CDataset
            Dataset to binarize.

        Returns
        -------
        bin_dataset : CDataset
            Binarized dataset.

        """
        raise NotImplementedError

    def apply_method(self, method, *args, **kwargs):
        """Apply input method to all trained classifiers.

        Useful to perform a routine after training (e.g. reduction, optim)

        `method` is an unbound method to apply, e.g. CClassiferSVM.set
        Any other argument for `method` can be passed in.

        """
        # Applying method to all trained classifiers
        for clf in self._binary_classifiers:
            # Unbound method: First argument is the instance to apply method to
            method(clf, *args, **kwargs)
