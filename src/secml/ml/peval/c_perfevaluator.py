"""
.. module:: PerformanceEvaluation
   :synopsis: Common interface and methods for performance estimation

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

from secml.core import CCreator
from secml.array import CArray
from secml.data.splitter import CDataSplitter
from secml.ml.peval.metrics import CMetric
from secml.parallel import parfor2


def _evaluate_one(
        row_id, perf_eval, params, params_matrix, estimator, ds, verbose):
    """Evaluate performance of estimator for one combination of parameters.

    Parameters
    ----------
    row_id : int
        Index of the row of params_matrix from which parameters
        to test should be extracted.
    perf_eval : CPerfEvaluator
        Evaluator object that will be used for performance evaluation.
    params : dict
        Dictionary with the parameters to be evaluated.
    params_matrix : CArray
        Indices of each combination of parameters to evaluate.
    estimator : CClassifier
        The classifier for witch we want chose best parameters.
    ds : CDataset
        Dataset to be used for evaluating parameters.
    verbose : int
        Sets verbosity level of the performance evaluator object.

    """
    # Build a dictionary with parameters to evaluate
    estimator_params = {}
    for par_idx, par in enumerate(params):
        # This works as params is an OrderedDict
        value_id = params_matrix[row_id, par_idx].item()
        estimator_params[par] = params[par][value_id]

    # Set estimator parameters using current combination
    estimator.set_params(estimator_params)

    # Resetting verbosity level as parallel copy the object
    perf_eval.verbose = verbose

    # Compute performance for current params set
    eval_score = perf_eval.compute_performance(estimator, ds)

    perf_eval.logger.info(
        "Params: {:} - Score: {:}".format(estimator_params, eval_score))

    return eval_score


class CPerfEvaluator(CCreator, metaclass=ABCMeta):
    """Evaluate the best parameters for input estimator.

    Parameters
    ----------
    splitter : CDataSplitter or str
        Object to use for splitting the dataset into train and validation.
    metric : CMetric or str
        Name of the metric that we want maximize / minimize.

    """
    __super__ = 'CPerfEvaluator'

    def __init__(self, splitter, metric):

        self.splitter = CDataSplitter.create(splitter)
        self.metric = CMetric.create(metric)

    def evaluate_params(
            self, estimator, dataset, parameters, pick='first', n_jobs=1):
        """Evaluate parameters for input estimator on input dataset.

        Parameters
        ----------
        estimator : CClassifier
            The classifier for witch we want chose best parameters.
        dataset : CDataset
            Dataset to be used for evaluating parameters.
        parameters : dict
            Dictionary with each entry as {parameter: list of values to test}.
        pick : {'first', 'last', 'random'}, optional
            Defines which of the best parameters set pick.
            Usually, 'first' (default) correspond to the smallest
            parameters while 'last' correspond to the biggest.
            The order is consistent to the parameters dict passed as input.
        n_jobs : int, optional
            Number of parallel workers to use. Default 1.
            Cannot be higher than processor's number of cores.

        Returns
        -------
        best_param_dict : dict
            A dictionary with the best value for each evaluated parameter.
        best_value : any
            Metric value obtained on validation set by the estimator.

        """
        self.logger.info("Parameters to evaluate: {:}".format(parameters))

        # FIRST OF ALL: save current classifier to restore later
        original_estimator = deepcopy(estimator)

        # Compute dataset splits
        self.splitter.compute_indices(dataset)

        # OrderedDict returns keys always in the same order,
        # so we are safe when iterating on params_matrix.shape[1]
        parameters = OrderedDict(
            sorted(parameters.items(), key=lambda t: t[0]))

        params_idx = []
        # create a list of list 'param_idx' with index of parameters' values
        for param_name in parameters:
            if not isinstance(parameters[param_name], list):
                raise TypeError("values for parameter `{:}` must be "
                                "specified as a list.".format(param_name))
            # Add an index for each parameter's value
            params_idx.append(list(range(len(parameters[param_name]))))

        # this is a matrix of indices.... e.g. [[1,1] [1,2], ..]
        # each row corresponds to the indices of parameters to be set
        params_matrix = CArray.comblist(params_idx).astype(int)

        # Parallelize (if requested) over the rows of params_matrix
        res_vect = parfor2(_evaluate_one, params_matrix.shape[0],
                           n_jobs, self, parameters, params_matrix,
                           estimator, dataset, self.verbose)
        # Transforming the list to array
        res_vect = CArray(res_vect)

        # Retrieve the best parameters
        best_params_dict, best_value = self._get_best_params(
            res_vect, parameters, params_matrix, pick=pick)

        self.logger.info("Best params: {:} - Value: {:}".format(
            best_params_dict, best_value))

        # Restore original parameters of classifier
        for param in original_estimator.__dict__:
            estimator.__dict__[param] = original_estimator.__dict__[param]

        return best_params_dict, best_value

    @abstractmethod
    def compute_performance(self, estimator, dataset):
        """Compute estimator performance on input dataset.

        This must be reimplemented by subclasses.

        Parameters
        ----------
        estimator : CClassifier 
            The classifier that we want evaluate.
        dataset : CDataset
            Dataset that we want use for evaluate the classifier.
        
        Returns
        -------        
        score : float
            Performance score of estimator.

        """
        raise NotImplementedError()

    @abstractmethod
    def _get_best_params(self, res_vect, params, params_matrix, pick='first'):
        """Returns the best parameters given input performance data.

        Parameters
        ----------
        res_vect : CArray
            Array with the performance results associated
            to each parameters combination.
        params : dict
            Dictionary with the parameters to be evaluated.
        params_matrix : CArray
            Indices of each combination of parameters to evaluate.
        pick : {'first', 'last', 'random'}, optional
            Defines which of the best parameters set pick.
            Usually, 'first' (default) correspond to the smallest
            parameters while 'last' correspond to the biggest.
            The order is consistent to the parameters dict passed as input.

        Returns
        -------
        best_params_dict : dict
            Dictionary with the parameters that have obtained
            the best performance score.
        best_value : any
            Performance value associated with the best parameters.

        """
        raise NotImplementedError()
