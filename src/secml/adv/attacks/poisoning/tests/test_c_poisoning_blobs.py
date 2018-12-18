from secml.utils import CUnitTest
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CDataSplitterShuffle
from secml.ml.features.normalization import CNormalizerMinMax
from test_c_poisoning import CPoisoningTestCases
from abc import abstractmethod

class TestCPoisoningBlob(CPoisoningTestCases.TestCPoisoning):

    @abstractmethod
    def param_setter(self):
        raise NotImplemented

    def _dataset_creation(self):
        self.n_features = 2  # Number of dataset features

        self.n_tr = 50
        self.n_ts = 1000
        self.n_classes = 2

        # Random state generator for the dataset
        self.seed = 44

        if self.n_classes == 2:
            loader = CDLRandomBlobs(
                n_samples=self.n_tr + self.n_ts,
                n_features=self.n_features,
                centers=[(-1, -1), (+1, +1)],
                center_box=(-2, 2),
                cluster_std=0.8,
                random_state=self.seed)

        self.logger.info(
            "Loading `random_blobs` with seed: {:}".format(self.seed))

        dataset = loader.load()
        splitter = CDataSplitterShuffle(num_folds=1, train_size=self.n_tr,
                                        random_state=3)
        splitter.compute_indices(dataset)
        self.tr = dataset[splitter.tr_idx[0], :]
        self.ts = dataset[splitter.ts_idx[0], :]

        normalizer = CNormalizerMinMax(feature_range=(-1, 1))
        self.tr.X = normalizer.train_normalize(self.tr.X)
        self.ts.X = normalizer.normalize(self.ts.X)

        self.lb = -1
        self.ub = 1

        self.grid_limits = [(self.lb - 0.1, self.ub + 0.1),
                            (self.lb - 0.1, self.ub + 0.1)]

class TestCPoisoningBlobRidge(TestCPoisoningBlob):
    def param_setter(self):
        self.clf_idx = 'ridge'

class TestCPoisoningBlobLogistic(TestCPoisoningBlob):
    def param_setter(self):
        self.clf_idx = 'logistic'

if __name__ == '__main__':
    CUnitTest.main()