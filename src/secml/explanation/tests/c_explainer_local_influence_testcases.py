from secml.utils import CUnitTest

from secml.explanation import CExplainerLocalInfluence
from secml.array import CArray
from secml.data.loader import CDataLoaderMNIST
from secml.data.splitter import CDataSplitterKFold
from secml.ml.peval.metrics import CMetricAccuracy
from secml.ml.classifiers.gradients.tests.utils import CClassifierGradientTest


class CExplainerLocalInfluenceTestCases(CUnitTest):
    """Unittests interface for CExplainerLocalInfluence."""

    def _create_mnist_dataset(self, digits=[4, 9], n_tr=50, n_val=1000,
                              n_ts=1000,
                              seed=10):
        loader = CDataLoaderMNIST()

        tr = loader.load('training', digits=digits)

        ts = loader.load('testing', digits=digits, num_samples=n_ts)

        # start train and validation dataset split
        splitter = CDataSplitterKFold(num_folds=2, random_state=seed)
        splitter.compute_indices(tr)

        val_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples),
                                        n_val,
                                        random_state=seed)
        val = tr[val_dts_idx, :]

        tr_dts_idx = CArray.randsample(CArray.arange(0, tr.num_samples), n_tr,
                                       random_state=seed)
        tr = tr[tr_dts_idx, :]

        tr.X /= 255.0
        val.X /= 255.0
        ts.X /= 255.0

        return tr, val, ts

    def setUp(self):
        self._tr, self._val, self._ts = self._create_mnist_dataset()

        self._clf_creation()
        self._clf.fit(self._tr)

        clf_loss = self._clf.gradients._loss.class_type
        explanation = CExplainerLocalInfluence(self._clf, self._tr,
                                               outer_loss_idx=clf_loss)

        self.influences = explanation.explain(self._ts.X, self._ts.Y)

        self.clf_gradients = CClassifierGradientTest.create(
            self._clf.class_type, self._clf.gradients)

    def _get_tr_without_point(self, p_idx):
        """
        Given the idx of a point return a copy of the training dataset
        without that point

        Parameters
        ----------
        p_idx int
            idx of the point that is wanted to be excluded by the training
            dataset

        Returns
        -------
        new_tr CDataset
                dataset without the point with the given index
        """
        all_idx = CArray.arange(self._tr.num_samples)
        not_p_idx = all_idx.find(all_idx != p_idx)
        new_tr = self._tr[not_p_idx, :]
        return new_tr

    def _check_influence(self, point_idx):
        """
        This function learn the classifier without a point and return the
        classifier loss on the test set.

        Parameters
        ----------
        point CArray
            training sample
        """
        clf_copy = self._clf.deepcopy()
        new_dataset = self._get_tr_without_point(point_idx)
        clf_copy.fit(new_dataset)

        loss = (1.0 / self._ts.num_samples) * self.clf_gradients.L(self._ts.X,
                                                                self._ts.Y,
                                                                clf_copy,
                                                                regularized=False).sum(
            axis=None)

        return loss

    def _check_prototype_pair(self, p_inf_idx, p_not_inf_idx):
        """
        Given a pair of prototypes where one is supposed to be between the
        most influent and the one between the less influent check if this is true.

        Parameters
        ----------
        p_inf_idx integer
            index in the training set of a sample supposed to be between the
            most influent
        p_not_inf_idx integer
            index in the training set of a sample supposed to be between the
            less influent
        """
        acc_without_p_infl = self._check_influence(p_inf_idx)
        self.logger.info("The loss without the point {:} supposed to be "
                         "one of the most influent is {:}".format(p_inf_idx,
                                                                  acc_without_p_infl))
        acc_without_p_not_infl = self._check_influence(p_not_inf_idx)
        self.logger.info("The loss without the point {:} supposed to be "
                         "one of the less influent is {:}".format(
            p_not_inf_idx,
            acc_without_p_not_infl))

        self.assertGreater(acc_without_p_infl, acc_without_p_not_infl,
                        "The point that is supposed to be between the "
                        "less influent has a higher influence of the "
                        "point supposed to be between one of the most "
                        "influent")

    def _test_explanation(self):
        self.assertEqual(self.influences.shape,
                         (self._ts.num_samples, self._tr.num_samples),
                         "The shape of the influences is wrong!")

        average_influence = self.influences.mean(axis=0).ravel()
        # order the idx of the tr samples in the way to have the less
        # influent in the first position of the vector and the most influent
        # in the las ones
        avg_infl_idx = average_influence.argsort()

        n_check = 5
        for i in xrange(1, n_check + 1):
            not_infl_idx = avg_infl_idx[i - 1].item()
            infl_idx = avg_infl_idx[-i].item()
            self._check_prototype_pair(infl_idx, not_infl_idx)


if __name__ == '__main__':
    CUnitTest.main()
