"""
.. module:: ROC
   :synopsis: Receiver Operating Characteristic (ROC) Curve

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@diee.unica.it>

"""
import sklearn.metrics as skm

from prlib.array import CArray


def refine_roc(tp, fp, th):
    """Function to ensure the bounds of a ROC.

    The first and last points should be (0,0) and (1,1) respectively.

    Parameters
    ----------
    tp : CArray
        True Positives, as returned by `.BaseRoc.compute()`
    fp : CArray
        False Positives, as returned by `.BaseRoc.compute()`
    th : CArray
        Thresholds, as returned by `.BaseRoc.compute()`

    """
    if tp[0] != fp[0] or tp[0] != 0 or fp[0] != 0:
        fp = CArray(0).append(fp)
        tp = CArray(0).append(tp)
        th = CArray(th[0] + 0.1).append(th)
    if tp[-1] != fp[-1] or tp[-1] != 1 or fp[-1] != 1:
        fp = fp.append(1)
        tp = tp.append(1)
        th = th.append(th[-1] - 0.1)
    return tp, fp, th


def average(fp, tp, n_points=1000):
    """Compute the average of the input tp/fp pairs.

    Parameters
    ----------
    fp, tp : CArray or list of CArray
        CArray or list of CArrays with false/true
        positives as output of `.CRoc`.
    n_points : int, optional
        Default 1000, is the number of points to be used for interpolation.

    Returns
    -------
    mean_fp : CArray
        Flat array with increasing false positive rates averaged over all
        available repetitions. Element i is the false positive rate of
        predictions with score >= thresholds[i].
    mean_tp : CArray
        Flat array with increasing true positive rates averaged over all
        available repetitions. Element i is the true positive rate of
        predictions with score >= thresholds[i].
    std_dev_tp : CArray
        Flat array with standard deviation of True Positives.

    """
    # Working with lists
    fp_list = [fp] if not isinstance(fp, list) else fp
    tp_list = [tp] if not isinstance(tp, list) else tp

    n_fp = len(fp_list)
    n_tp = len(tp_list)

    # Checking consistency between input data
    if n_fp == 0:
        raise ValueError("At least 1 array with false/true "
                         "positives must be specified.")
    if n_fp != n_tp:
        raise ValueError("Number of true positives and false "
                         "positives must be the same.")

    # Computing ROC for a single (labels, scores) pair
    mean_fp = CArray.linspace(0, 1, n_points)
    mean_tp = 0.0

    all_roc_tp = CArray.zeros(shape=(n_tp, n_points))

    for i, data_i in enumerate(zip(fp_list, tp_list)):
        # Interpolating over 'x' axis
        i_tp = mean_fp.interp(*data_i)
        # Will be used later to compute std
        all_roc_tp[i, :] = i_tp
        # Adding current tp to mean_tp
        mean_tp += i_tp
        mean_tp[0] = 0.0  # First point should be (0,0) to prevent side effects

    mean_tp /= n_tp
    mean_tp[-1] = 1.0  # Last point should be (1,1) to prevent side effects

    # Computing standard deviation
    std_dev_tp = all_roc_tp.std(axis=0, keepdims=False)
    std_dev_tp[-1] = 0

    return mean_fp, mean_tp, std_dev_tp


class CBaseRoc(object):
    """Computes the receiver operating characteristic curve, or ROC curve.

    This base class manage a single classifier output (a single repetition).

    See Also
    --------
    .CRoc : class that fully supports ROC repetitions.

    """
    def __init__(self):
        self._fp = None
        self._tp = None
        self._th = None

    @property
    def fp(self):
        """False Positives.

        Flat array with increasing false positive rates. Element i
        is the false positive rate of predictions with score >= thresholds[i].

        """
        return self._fp

    @property
    def tp(self):
        """True Positives.

        Flat array with increasing true positive rates. Element i
        is the true positive rate of predictions with score >= thresholds[i].

        """
        return self._tp

    @property
    def th(self):
        """Thresholds.

        Flat array with decreasing thresholds on the decision function
        used to compute fpr and tpr. `thresholds[0]` represents no
        instances being predicted and is arbitrarily set to `max(score) + 0.1`.

        """
        return self._th

    def compute(self, y_true, score, positive_label=None):
        """Compute TP/FP for classifier output.

        Parameters
        ----------
        y_true : CArray
            Flat array with true binary labels in range {0, 1} or {-1, 1}
            for each patterns or a single array.
            If labels are not binary, pos_label should be explicitly given.
        score : CArray
            Flat array with target scores for each pattern, can either be
            probability estimates of the positive class or confidence values.
        positive_label : int, optional
            Label to consider as positive (others are considered negative).

        Returns
        -------
        single_roc : CBaseRoc
            Instance of the roc curve (tp, fp, th).

        """
        # Computing ROC for a single (labels, scores) pair
        fp, tp, th = skm.roc_curve(CArray(y_true).tondarray().ravel(),
                                   CArray(score).tondarray().ravel(),
                                   positive_label)
        fp = CArray(fp)
        tp = CArray(tp)
        th = CArray(th)
        # Ensure first and last points are (0,0) and (1,1) respectively
        self._fp, self._tp, self._th = refine_roc(fp, tp, th)

        return self

    def __iter__(self):
        """Returns `fp`, `tp`, `th` always in this order."""
        seq = ('fp', 'tp', 'th')  # Fixed order for consistency
        for e in seq:
            yield getattr(self, e)

    def reset(self):
        """Reset stored data."""
        self._fp = None
        self._tp = None
        self._th = None


class CRoc(CBaseRoc):
    """Computes the receiver operating characteristic curve, or ROC curve.

        "A receiver operating characteristic (ROC), or simply ROC curve,
        is a graphical plot which illustrates the performance of a binary
        classifier system as its discrimination threshold is varied.
        It is created by plotting the fraction of true positives out of
        the positives (TPR = true positive rate) vs. the fraction of false
        positives out of the negatives (FPR = false positive rate),
        at various threshold settings. TPR is also known as sensitivity,
        and FPR is one minus the specificity or true negative rate."

    The class manage different repetitions of the same classification output.

    """

    def __init__(self):
        # Calling superclass constructor
        super(CRoc, self).__init__()
        # Output structures
        self._data = []
        self._data_average = CBaseRoc()
        self._std_dev_tp = None

    @property
    def fp(self):
        """False Positives.

        Flat array with increasing false positive rates or a list with
        one array for each repetition. Element i is the false positive
        rate of predictions with score >= thresholds[i].

        """
        # This returns a list or a single arrays if one rep is available
        fp = map(list, zip(*self._data))[0]
        return fp[0] if len(fp) == 1 else fp

    @property
    def tp(self):
        """True Positives.

        Flat array with increasing true positive rates or a list with
        one array for each repetition. Element i is the true positive
        rate of predictions with score >= thresholds[i].

        """
        # This returns a list or a single arrays if one rep is available
        tp = map(list, zip(*self._data))[1]
        return tp[0] if len(tp) == 1 else tp

    @property
    def th(self):
        """Thresholds.

        Flat array with decreasing thresholds on the decision function
        used to compute fpr and tpr or a list with one array for each
        repetition. `thresholds[0]` represents no instances being
        predicted and is arbitrarily set to `max(score) + 0.1`.

        """
        # This returns a list or a single arrays if one rep is available
        th = map(list, zip(*self._data))[2]
        return th[0] if len(th) == 1 else th

    @property
    def n_reps(self):
        """Return the number of computed ROC."""
        return len(self._data)

    @property
    def has_mean(self):
        """True if average has been computed for all ROCs."""
        return False if self.mean_fp is None or self.mean_tp is None else True

    @property
    def has_std_dev(self):
        """True if standard deviation has been computed for all ROCs."""
        return False if self._std_dev_tp is None else True

    @property
    def mean_fp(self):
        """Averaged False Positives.

        Flat array with increasing false positive rates averaged over all
        available repetitions. Element i is the false positive rate of
        predictions with score >= thresholds[i].

        """
        return self._data_average.fp

    @property
    def mean_tp(self):
        """Averaged True Positives.

        Flat array with increasing true positive rates averaged over all
        available repetitions. Element i is the true positive rate of
        predictions with score >= thresholds[i].

        """
        return self._data_average.tp

    @property
    def std_dev_tp(self):
        """Standard deviation of True Positives."""
        return self._std_dev_tp

    def compute(self, y_true, score, positive_label=None):
        """Compute ROC curve using input True labels and Classification Scores.

        For multi-class data, label to be considered positive should specified.

        If `y_true` and `score` are both lists (with same length),
        one roc curve for each pair is returned.
        If `y_true` is a single array, one roc curve for each
        (y_true, score[i]) is returned.

        Each time the function is called, result is appended to
        `tp`,`fp`, and `thr` class attributes.
        Returned ROCs are the only associated with LATEST input data.

        Parameters
        ----------
        y_true : CArray, list
            List of flat arrays with true binary labels in range
            {0, 1} or {-1, 1} for each patterns or a single array.
            If a single array, one curve is returned
            for each (y_true, score[i]) pair.
            If labels are not binary, pos_label should be explicitly given.
        score : CArray, list
            List of flat array with target scores for each pattern,
            can either be probability estimates of the positive
            class or confidence values.
            If `y_true` is a single array, one curve is returned for each
            (y_true, score[i]) pair.
        positive_label : int, optional
            Label to consider as positive (others are considered negative).

        Returns
        -------
        fp : CArray or list
            Flat array with increasing false positive rates or a list with
            one array for each repetition. Element i is the false positive
            rate of predictions with score >= thresholds[i]
        tp : CArray or list
            Flat array with increasing true positive rates or a list with
            one array for each repetition. Element i is the true positive
            rate of predictions with score >= thresholds[i].
        th : CArray or list
            Flat array with decreasing thresholds on the decision function
            used to compute fpr and tpr or a list with one array for each
            repetition. `thresholds[0]` represents no instances being
            predicted and is arbitrarily set to `max(score) + 0.1`.

        """
        # Working with lists
        y_true_list = [y_true] if not isinstance(y_true, list) else y_true
        score_list = [score] if not isinstance(score, list) else score

        n_ytrue = len(y_true_list)
        n_score = len(score_list)

        # Checking consistency between input data
        if n_score == 0:
            raise ValueError("At least 1 array with classification "
                             "scores must be specified.")
        if n_ytrue != n_score and n_ytrue + n_score != n_score + 1:
            raise ValueError("Either 1 or {:} labels arrays should "
                             "be specified.".format(n_score))

        # Resetting any computed average ROC
        self._data_average.reset()
        self._std_dev_tp = None

        if n_ytrue == 1:  # Use the same true labels vs all scores
            for score_idx in xrange(n_score):
                rep = CBaseRoc().compute(y_true_list[0],
                                         score_list[score_idx],
                                         positive_label)
                # Storing result as a new repetition for ROC
                self._data.append(rep)

        else:  # Use each true labels vs corresponding scores
            for score_idx in xrange(n_score):
                rep = CBaseRoc().compute(y_true_list[score_idx],
                                         score_list[score_idx],
                                         positive_label)
                # Storing result as a new repetition for ROC
                self._data.append(rep)

        out = []
        # Some hardcore python next: this returns 3 separate lists
        # (fp, tp, thr) or 3 single arrays if one repetition is available
        for e in map(list, zip(*self._data[-n_score:])):
            out.append(e[0] if len(e) == 1 else e)

        return tuple(out)

    def average(self, n_points=1000, return_std=False):
        """Compute the average of computed ROC curves.

        The average ROC is reset each time `.compute_roc` is called.

        Parameters
        ----------
        n_points : int, optional
            Default 1000, is the number of points to be used for interpolation.
        return_std : bool, optional
            If True, standard deviation of True Positives will be returned.

        Returns
        -------
        mean_fp : CArray
            Flat array with increasing false positive rates averaged over all
            available repetitions. Element i is the false positive rate of
            predictions with score >= thresholds[i].
        mean_tp : CArray
            Flat array with increasing true positive rates averaged over all
            available repetitions. Element i is the true positive rate of
            predictions with score >= thresholds[i].
        std_dev_tp : CArray
            Flat array with standard deviation of True Positives.
            Only if return_std is True.

        """
        mean_fp, mean_tp, std_dev_tp = average(self.fp, self.tp,
                                               n_points=n_points)

        # Assigning final data
        self._data_average._fp = mean_fp
        self._data_average._tp = mean_tp
        self._std_dev_tp = std_dev_tp

        out = tuple(self._data_average)[0:2]
        if return_std is True:  # Return standard deviation if needed
            out += (self._std_dev_tp, )
        return out
