"""
.. module:: PerformanceEvaluationXVal
   :synopsis: Best parameters estimation with Cross-Validation

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.ml.peval import CPerfEvaluator
from secml.array import CArray
from secml.core.type_utils import is_scalar


class CPerfEvaluatorXVal(CPerfEvaluator):
    """Evaluate the best estimator parameters using Cross-Validation.

    Parameters
    ----------
    splitter : CXVal or str
        XVal object to be used for splitting the dataset
        into train and validation.
    metric : CMetric or str
        Name of the metric that we want maximize / minimize.

    Attributes
    ----------
    class_type : 'xval'

    """
    __class_type = 'xval'

    def compute_performance(self, estimator, dataset):
        """Split data in folds and return the mean estimator performance.

        Parameters
        ----------
        estimator : CClassifier 
            The Classifier that we want evaluate
        dataset : CDataset
            Dataset that we want use for evaluate the classifier

        Returns
        -------        
        score : float
            Mean performance score of estimator computed on the K-Folds.

        """
        # Placeholder for folds' score
        fold_number = len(self.splitter.tr_idx)
        splits_score = CArray.zeros(fold_number)

        # estimate the performance of the estimator on each fold
        for split_idx in range(fold_number):

            train_dataset = dataset[self.splitter.tr_idx[split_idx], :]
            test_dataset = dataset[self.splitter.ts_idx[split_idx], :]

            # Train the estimator
            estimator.fit(train_dataset.X, train_dataset.Y)

            pred_label, pred_score = estimator.predict(
                test_dataset.X, return_decision_function=True)

            if dataset.num_classes > 2:
                pred_score = None  # Score cannot be used in multiclass case
            else:
                # Extracting score of the positive class
                pred_score = pred_score[:, 1].ravel()

            this_test_score = self.metric.performance_score(
                test_dataset.Y, y_pred=pred_label, score=pred_score)

            splits_score[split_idx] = this_test_score

        return splits_score.mean()

    def _get_best_params(self, res_vect, params, params_matrix, pick='first'):
        """Returns the best parameters given input performance scores.

        The best parameters have the closest associated performance score
        to the metric's best value.

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
        if not is_scalar(self.metric.best_value):
            raise TypeError(
                "XVal only works with metric with the best value as scalar")

        # Get the index of the results closest to the best value
        diff = abs(res_vect - self.metric.best_value)
        condidates_idx = diff.find(diff == diff.nanmin())

        if len(condidates_idx) < 1:
            raise ValueError("all metric outputs are equal to Nan!")

        # Get the value of the result closest to the best value
        best_score = res_vect[condidates_idx[0]]

        # Get the index of the corresponding parameters
        best_params_idx = params_matrix[condidates_idx, :]

        # Build the list of candidate parameters
        best_params_list = []
        for c_idx in range(best_params_idx.shape[0]):
            # For each candidate get corresponding parameters
            best_params_dict = dict()
            for j, par in enumerate(params):
                value_idx = best_params_idx[c_idx, j].item()
                best_params_dict[par] = params[par][value_idx]

            best_params_list.append(best_params_dict)

        # Chose which candidate parameters assign to classifier
        if pick == 'first':  # Usually the smallest
            best_params_dict = best_params_list[0]
        elif pick == 'last':  # Usually the biggest
            best_params_dict = best_params_list[-1]
        elif pick == 'random':
            import random
            best_params_dict = random.choice(best_params_list)
        else:
            raise ValueError("pick strategy '{:}' not known".format(pick))

        return best_params_dict, best_score
