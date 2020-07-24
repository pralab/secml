"""
.. module:: PerformanceEvaluationXValMulticlass
   :synopsis: Best parameters estimation with Cross-Validation for multiclass

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
from secml.ml.peval import CPerfEvaluator
from secml.array import CArray
from secml.core.type_utils import is_scalar


class CPerfEvaluatorXValMulticlass(CPerfEvaluator):
    """Evaluate the best parameters for each single binary classifier using Cross-Validation.

    Parameters
    ----------
    splitter : CXVal or str
        XVal object to be used for splitting the dataset
        into train and validation.
    metric : CMetric or str
        Name of the metric that we want maximize / minimize.

    Attributes
    ----------
    class_type : 'xval-multiclass'

    """
    __class_type = 'xval-multiclass'

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
        scores : list
            Mean performance score of each binary estimator
            computed on the K-Folds.

        """
        # Placeholder for folds' score
        num_folds = len(self.splitter.tr_idx)
        test_scores = CArray.zeros(shape=(num_folds, dataset.num_classes))

        # estimate the performance of the estimator on each fold
        for split_idx in range(num_folds):

            train_dataset = dataset[self.splitter.tr_idx[split_idx], :]
            test_dataset = dataset[self.splitter.ts_idx[split_idx], :]

            # Fit the estimator
            estimator.fit(train_dataset.X, train_dataset.Y)

            # Get the classification performance of each binary estimator
            split_scores = []
            for class_idx in range(dataset.num_classes):
                # Binarize dataset
                test_binary_ds = estimator.binarize_dataset(
                    class_idx, test_dataset)
                # Extract the target internal binary estimator.
                # They are all trained on the same data (normalized if needed)
                binary_clf = estimator._binary_classifiers[class_idx]

                pred_label, pred_score = binary_clf.predict(
                    test_binary_ds.X, return_decision_function=True)

                # Extracting score of the positive class
                pred_score = pred_score[:, 1].ravel()

                this_test_score = self.metric.performance_score(
                    test_binary_ds.Y, y_pred=pred_label, score=pred_score)
                split_scores.append(this_test_score)

            test_scores[split_idx, :] = CArray(split_scores)

        return test_scores.mean(axis=0, keepdims=False).tolist()

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

        best_params_list = []
        best_score = []
        # Get the best parameters for each binary classifier
        for i in range(res_vect.shape[1]):

            # diff has one row for each parameters combination and
            # one column for each binary classifier
            condidates_idx = diff[:, i].find_2d(
                diff[:, i] == diff[:, i].min())[0]

            # Get the value of the result closest to the best value
            best_score.append(res_vect[condidates_idx[0], i])

            # Get the index of the corresponding parameters
            best_params_idx = params_matrix[condidates_idx, :]

            # Build the list of candidate parameters for binary clf
            clf_best_params_list = []
            for c_idx in range(best_params_idx.shape[0]):
                # For each candidate get corresponding parameters
                best_params_dict = dict()
                for j, par in enumerate(params):
                    par_idx = best_params_idx[c_idx, j].item()
                    best_params_dict[par] = params[par][par_idx]

                clf_best_params_list.append(best_params_dict)

            # Chose which candidate parameters assign to classifier
            if pick == 'first':  # Usually the smallest
                clf_best_params_dict = clf_best_params_list[0]
            elif pick == 'last':  # Usually the biggest
                clf_best_params_dict = clf_best_params_list[-1]
            elif pick == 'random':
                import random
                clf_best_params_dict = random.choice(clf_best_params_list)
            else:
                raise ValueError("pick strategy '{:}' not known".format(pick))

            best_params_list.append(clf_best_params_dict)

        # For each param, built the tuple of the best value for each binary clf
        best_params_dict = dict()
        for par in params:
            this_param_list = []
            for params_dict in best_params_list:
                this_param_list.append(params_dict[par])
            best_params_dict[par] = tuple(this_param_list)

        return best_params_dict, best_score
