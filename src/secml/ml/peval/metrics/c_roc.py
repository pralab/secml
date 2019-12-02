"""
.. module:: CROC
   :synopsis: Receiver Operating Characteristic (ROC) Curve

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Ambra Demontis <ambra.demontis@unica.it>

"""
from secml.array import CArray


def refine_roc(fpr, tpr, th):
    """Function to ensure the bounds of a ROC.

    The first and last points should be (0,0) and (1,1) respectively.

    Parameters
    ----------
    fpr : CArray
        False Positive Rates, as returned by `.BaseRoc.compute()`.
    tpr : CArray
        True Positive Rates, as returned by `.BaseRoc.compute()`.
    th : CArray
        Thresholds, as returned by `.BaseRoc.compute()`.

    """
    if tpr[0] != fpr[0] or tpr[0] != 0 or fpr[0] != 0:
        fpr = CArray(0).append(fpr)
        tpr = CArray(0).append(tpr)
        th = CArray(th[0] + 1e-3).append(th)
    if tpr[-1] != fpr[-1] or tpr[-1] != 1 or fpr[-1] != 1:
        fpr = fpr.append(1)
        tpr = tpr.append(1)
        th = th.append(th[-1] - 1e-3)
    return fpr, tpr, th


def average(fpr, tpr, n_points=1000):
    """Compute the average of the input tpr/fpr pairs.

    Parameters
    ----------
    fpr, tpr : CArray or list of CArray
        CArray or list of CArrays with False/True Positive Rates
        as output of `.CRoc`.
    n_points : int, optional
        Default 1000, is the number of points to be used for interpolation.

    Returns
    -------
    mean_fpr : CArray
        Flat array with increasing False Positive Rates averaged over all
        available repetitions. Element i is the False Positive Rate of
        predictions with score >= thresholds[i].
    mean_tpr : CArray
        Flat array with increasing True Positive Rates averaged over all
        available repetitions. Element i is the True Positive Rate of
        predictions with score >= thresholds[i].
    std_dev_tpr : CArray
        Flat array with standard deviation of True Positive Rates.

    """
    # Working with lists
    fpr_list = [fpr] if not isinstance(fpr, list) else fpr
    tpr_list = [tpr] if not isinstance(tpr, list) else tpr

    n_fpr = len(fpr_list)
    n_tpr = len(tpr_list)

    # Checking consistency between input data
    if n_fpr == 0:
        raise ValueError("At least 1 array with false/true "
                         "positives must be specified.")
    if n_fpr != n_tpr:
        raise ValueError("Number of True Positive Rates and "
                         "False Positive Rates must be the same.")

    # Computing ROC for a single (labels, scores) pair
    mean_fpr = CArray.linspace(0, 1, n_points)
    mean_tpr = 0.0

    all_roc_tpr = CArray.zeros(shape=(n_tpr, n_points))

    for i, data_i in enumerate(zip(fpr_list, tpr_list)):
        # Interpolating over 'x' axis
        i_tpr = mean_fpr.interp(*data_i)
        # Will be used later to compute std
        all_roc_tpr[i, :] = i_tpr
        # Adding current tpr to mean_tpr
        mean_tpr += i_tpr
        mean_tpr[0] = 0.0  # First should be (0,0) to prevent side effects

    mean_tpr /= n_tpr
    mean_tpr[-1] = 1.0  # Last point should be (1,1) to prevent side effects

    # Computing standard deviation
    std_dev_tpr = all_roc_tpr.std(axis=0, keepdims=False)
    std_dev_tpr[-1] = 0

    return mean_fpr, mean_tpr, std_dev_tpr


class CBaseRoc:
    """Computes the receiver operating characteristic curve, or ROC curve.

    This base class manage a single classifier output (a single repetition).

    See Also
    --------
    .CRoc : class that fully supports ROC repetitions.

    """
    def __init__(self):
        self._fpr = None
        self._tpr = None
        self._th = None

    @property
    def fpr(self):
        """False Positive Rates.

        Flat array with increasing False Positive Rates. Element i
         is the False Positive Rate of predictions with score >= thresholds[i].

        """
        return self._fpr

    @property
    def tpr(self):
        """True Positive Rates.

        Flat array with increasing True Positive Rates. Element i
         is the True Positive Rate of predictions with score >= thresholds[i].

        """
        return self._tpr

    @property
    def th(self):
        """Thresholds.

        Flat array with decreasing thresholds on the decision function
         used to compute fpr and tpr. `thresholds[0]` represents no
         instances being predicted and is arbitrarily set to
         `max(score) + 1e-3`.

        """
        return self._th

    def compute(self, y_true, score, positive_label=None):
        """Compute TPR/FPR for classifier output.

        Parameters
        ----------
        y_true : CArray
            Flat array with true binary labels in range {0, 1}
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
            Instance of the roc curve (tpr, fpr, th).

        """
        th = score.unique()  # unique also sorts the values

        n = CArray(score[y_true == 0])
        p = CArray(score[y_true == 1])

        # Counting the fpr and the tpr
        fp_list = []
        tp_list = []
        for i in range(th.size):
            fp_i = (n >= th[i]).sum() if n.size != 0 else 0
            tp_i = (p >= th[i]).sum() if p.size != 0 else 0
            fp_list.append(fp_i)
            tp_list.append(tp_i)

        # Returning increasing fpr, tpr...
        fp_list.reverse()
        tp_list.reverse()
        # ...and th accordingly (decreasing)
        th = CArray(th[::-1])

        # Normalizing in 0-1
        fpr = CArray(fp_list) / n.size if n.size != 0 else CArray([0])
        tpr = CArray(tp_list) / p.size if p.size != 0 else CArray([0])

        # Ensure first and last points are (0,0) and (1,1) respectively
        self._fpr, self._tpr, self._th = refine_roc(fpr, tpr, th)

        return self

    def __iter__(self):
        """Returns `fpr`, `tpr`, `th` always in this order."""
        seq = ('fpr', 'tpr', 'th')  # Fixed order for consistency
        for e in seq:
            yield getattr(self, e)

    def reset(self):
        """Reset stored data."""
        self._fpr = None
        self._tpr = None
        self._th = None


class CRoc(CBaseRoc):
    """Computes the receiver operating characteristic curve, or ROC curve.

        "A receiver operating characteristic (ROC), or simply ROC curve,
        is a graphical plot which illustrates the performance of a binary
        classifier system as its discrimination threshold is varied.
        It is created by plotting the fraction of True Positive Rates out of
        the Positives (TPR = True Positive Rate) vs. the fraction of False
        Positives out of the Negatives (FPR = False Positive Rate),
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
        self._std_dev_tpr = None

    @property
    def fpr(self):
        """False Positive Rates.

        Flat array with increasing False Positive Rates or a list with
         one array for each repetition. Element i is the False Positive
         Rate of predictions with score >= thresholds[i].

        """
        # This returns a list or a single arrays if one rep is available
        fpr = list(map(list, zip(*self._data)))[0]
        return fpr[0] if len(fpr) == 1 else fpr

    @property
    def tpr(self):
        """True Positive Rates.

        Flat array with increasing True Positive Rates or a list with
         one array for each repetition. Element i is the True Positive
         Rate of predictions with score >= thresholds[i].

        """
        # This returns a list or a single arrays if one rep is available
        tpr = list(map(list, zip(*self._data)))[1]
        return tpr[0] if len(tpr) == 1 else tpr

    @property
    def th(self):
        """Thresholds.

        Flat array with decreasing thresholds on the decision function
         used to compute fpr and tpr or a list with one array for each
         repetition. `thresholds[0]` represents no instances being
         predicted and is arbitrarily set to `max(score) + 1e-3`.

        """
        # This returns a list or a single arrays if one rep is available
        th = list(map(list, zip(*self._data)))[2]
        return th[0] if len(th) == 1 else th

    @property
    def n_reps(self):
        """Return the number of computed ROC."""
        return len(self._data)

    @property
    def has_mean(self):
        """True if average has been computed for all ROCs."""
        return False if self.mean_fpr is None or self.mean_tpr is None else True

    @property
    def has_std_dev(self):
        """True if standard deviation has been computed for all ROCs."""
        return False if self._std_dev_tpr is None else True

    @property
    def mean_fpr(self):
        """Averaged False Positive Rates.

        Flat array with increasing False Positive Rates averaged over all
         available repetitions. Element i is the false positive rate of
         predictions with score >= thresholds[i].

        """
        return self._data_average.fpr

    @property
    def mean_tpr(self):
        """Averaged True Positive Rates.

        Flat array with increasing True Positive Rates averaged over all
         available repetitions. Element i is the True Positive Rate of
         predictions with score >= thresholds[i].

        """
        return self._data_average.tpr

    @property
    def std_dev_tpr(self):
        """Standard deviation of True Positive Rates."""
        return self._std_dev_tpr

    def compute(self, y_true, score, positive_label=None):
        """Compute ROC curve using input True labels and Classification Scores.

        For multi-class data, label to be considered positive should specified.

        If `y_true` and `score` are both lists (with same length),
        one roc curve for each pair is returned.
        If `y_true` is a single array, one roc curve for each
        (y_true, score[i]) is returned.

        Each time the function is called, result is appended to
        `tpr`,`fpr`, and `thr` class attributes.
        Returned ROCs are the only associated with LATEST input data.

        Parameters
        ----------
        y_true : CArray, list
            List of flat arrays with true binary labels in range
            {0, 1} for each patterns or a single array.
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
        fpr : CArray or list
            Flat array with increasing False Positive Rates or a list with
             one array for each repetition. Element i is the False Positive
             Rate of predictions with score >= thresholds[i]
        tpr : CArray or list
            Flat array with increasing True Positive Rates or a list with
             one array for each repetition. Element i is the True Positive
             Rate of predictions with score >= thresholds[i].
        th : CArray or list
            Flat array with decreasing thresholds on the decision function
             used to compute fpr and tpr or a list with one array for each
             repetition. `thresholds[0]` represents no instances being
             predicted and is arbitrarily set to `max(score) + 1e-3`.

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
        self._std_dev_tpr = None

        if n_ytrue == 1:  # Use the same true labels vs all scores
            for score_idx in range(n_score):
                rep = CBaseRoc().compute(y_true_list[0],
                                         score_list[score_idx],
                                         positive_label)
                # Storing result as a new repetition for ROC
                self._data.append(rep)

        else:  # Use each true labels vs corresponding scores
            for score_idx in range(n_score):
                rep = CBaseRoc().compute(y_true_list[score_idx],
                                         score_list[score_idx],
                                         positive_label)
                # Storing result as a new repetition for ROC
                self._data.append(rep)

        out = []
        # Some hardcore python next: this returns 3 separate lists
        # (fpr, tpr, thr) or 3 single arrays if one repetition is available
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
            If True, standard deviation of True Positive Rates will be returned.

        Returns
        -------
        mean_fpr : CArray
            Flat array with increasing False Positive Rates averaged over all
             available repetitions. Element i is the false positive rate of
             predictions with score >= thresholds[i].
        mean_tpr : CArray
            Flat array with increasing True Positive Rates averaged over all
             available repetitions. Element i is the true positive rate of
             predictions with score >= thresholds[i].
        std_dev_tpr : CArray
            Flat array with standard deviation of True Positive Rates.
            Only if return_std is True.

        """
        mean_fpr, mean_tpr, std_dev_tpr = average(
            self.fpr, self.tpr, n_points=n_points)

        # Assigning final data
        self._data_average._fpr = mean_fpr
        self._data_average._tpr = mean_tpr
        self._std_dev_tpr = std_dev_tpr

        out = tuple(self._data_average)[0:2]
        if return_std is True:  # Return standard deviation if needed
            out += (self._std_dev_tpr,)
        return out
