"""
Created on 08 mar 2016

@author: Davide Maiorca
@author: Ambra Demontis

This module tests the CSecEval class, which performs an attacks on
a dataset w.r.t increasing attack power
TODO: Add assertEquals statements to check the correctness of the test

"""
from abc import ABCMeta, abstractmethod

from secml.utils import CUnitTest
from secml.adv.seceval import CSecEval
from secml.data.loader import CDLRandomBlobs
from secml.figure import CFigure
from secml.ml.classifiers import CClassifierSVM
from secml.ml.features.normalization import CNormalizerMinMax


class CSecEvalTestCases(object):
    """
    Wrapper for TestCSecEval to make unittest.main() work correctly
    """

    class TestCSecEval(CUnitTest):
        """
        SecEval unittest
        """
        __metaclass__ = ABCMeta

        @abstractmethod
        def attack_params_setter(self):
            """
            Set attack method dependent parameters
            :return:
            """
            pass

        def setUp(self):
            """
            Basic set up for the CEvasionSecEval class.
            """
            self.logger.verbose = 0

            # classifier settings:

            self.classifier = CClassifierSVM(
                kernel='linear', C=1.0, grad_sampling=1.0)
            # self.classifier.gamma = 1

            #######################################

            # data parameters
            self.discrete = False

            self.lb = -1
            self.ub = +1

            ############################
            # load / split datain
            # self.n_tr = 500
            # self.n_ts =  10 # 50

            # self.tr, self.ts = self._load_data_mnist(self.n_tr, self.n_ts)
            ###################################
            # dataset generation:

            self.n_tr = 20 #100 # 50
            self.n_val = 20
            self.n_ts = 40
            self.n_features = 2
            self.n_clusters = 2

            # Random state generator for the dataset
            # self.seed = 476843149
            # self.seed = 382600436
            self.seed = 396290607
            # self.seed = random.randint(999999999)

            loader = CDLRandomBlobs(
                n_samples=self.n_tr,
                n_features=self.n_features,
                centers=[(-1, -1), (+1, +1)],
                center_box=(-2, 2),
                cluster_std=0.8,
                random_state=self.seed)

            self.logger.info(
                "Loading `random_blobs` with seed: {:}".format(self.seed))
            self.tr = loader.load()

            loader.n_samples = self.n_val
            self.val = loader.load()

            loader.n_samples = self.n_ts
            self.ts = loader.load()

            normalizer = CNormalizerMinMax(feature_range=(self.lb, self.ub))

            self.tr.X = normalizer.train_normalize(self.tr.X)
            self.val.X = normalizer.normalize(self.val.X)
            self.ts.X = normalizer.normalize(self.ts.X)

            # self.val = self.tr

            self.classifier.train(self.tr)
            ##############################################

            # attack parameters settings
            self.attack_params_setter()

            # set sec eval object
            self.sec_eval = CSecEval(
                attack=self.attack,
                param_name=self.param_name,
                param_values=self.param_values,
                )

        def _plot_sec_eval(self):
            # figure creation
            figure = CFigure(height=5, width=5,
                             title="Obj. function "
                                   "under increasing {:} values".format(
                                 self.sec_eval.sec_eval_data.param_name))

            # plot security evaluation
            figure.switch_sptype('sec_eval')
            figure.sp.plot_metric(self.sec_eval.sec_eval_data)

            figure.subplots_adjust()
            figure.show()

        def _plot_sec_eval_for_each_class(self):
            # figure creation
            figure = CFigure(height=5, width=5,
                             title="Obj. function "
                                   "under increasing {:} values".format(
                                 self.sec_eval.sec_eval_data.param_name))

            # plot security evaluation
            figure.switch_sptype('sec_eval')
            figure.sp.plot_metric_for_class(self.sec_eval.sec_eval_data)

            figure.subplots_adjust()
            figure.show()


        def _fobj_plot(self):
            # figure creation
            figure = CFigure(height=5, width=5,
                             title="Obj. function "
                                   "under increasing {:} values".format(
                                 self.sec_eval._sec_eval_data.param_name))

            figure.sp.plot(self.sec_eval._sec_eval_data.param_values, self.sec_eval._sec_eval_data.fobj)
            figure.sp.ylabel("Objective function")
            figure.sp.xlabel(self.sec_eval._sec_eval_data.param_name)

            figure.subplots_adjust()
            figure.show()

        def test_sec_eval(self):

            # evaluate classifier security
            self.sec_eval.run_sec_eval(self.ts)
            self._plot_sec_eval()

            self._plot_sec_eval_for_each_class()
