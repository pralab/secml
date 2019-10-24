from math import ceil

from secml.testing import CUnitTest

try:
    import torchvision
except ImportError:
    CUnitTest.importskip("torchvision")
else:
    from torchvision import transforms

from secml.array import CArray
from secml.data import CDataset
from secml.data.loader import CDataLoaderMNIST, CDLRandom
from secml.data.loader.c_dataloader_pytorch import CDataLoaderPyTorch
from secml.data.splitter import CTrainTestSplit
from secml.ml.features import CNormalizerMinMax


class TestCDataLoaderPytorch(CUnitTest):
    """Unittest for CDataLoaderPyTorch"""

    def setUp(self):
        self.n_classes = 3
        self.n_features = 5
        self.n_samples_tr = 1000  # number of training set samples
        self.n_samples_ts = 500  # number of testing set samples
        self.batch_size = 20

    def _dataset_creation_blobs(self):
        self.logger.info("\tTest dataset creation")
        # generate synthetic data
        dataset = CDLRandom(n_samples=self.n_samples_tr + self.n_samples_ts,
                            n_classes=self.n_classes,
                            n_features=self.n_features, n_redundant=0,
                            n_clusters_per_class=1,
                            class_sep=2, random_state=0).load()

        # Split in training and test
        splitter = CTrainTestSplit(
            train_size=self.n_samples_tr, test_size=self.n_samples_ts,
            random_state=0)
        self.tr, self.ts = splitter.split(dataset)

        # Normalize the data
        nmz = CNormalizerMinMax()
        self.tr.X = nmz.fit_transform(self.tr.X)
        self.ts.X = nmz.transform(self.ts.X)

        self._tr_loader = CDataLoaderPyTorch(self.tr.X, self.tr.Y,
                                             self.batch_size, shuffle=True,
                                             transform=None).get_loader()

        self._ts_loader = CDataLoaderPyTorch(self.ts.X, self.ts.Y,
                                             self.batch_size, shuffle=False,
                                             transform=None).get_loader()

    def _dataset_creation_mnist(self):
        self.logger.info("\tTest dataset creation")
        digits = (1, 7)
        dataset = CDataLoaderMNIST().load('training', digits=digits)

        # Split in training and test
        splitter = CTrainTestSplit(
            train_size=self.n_samples_tr, test_size=self.n_samples_ts,
            random_state=0)
        self.tr, self.ts = splitter.split(dataset)

        # Normalize the data
        nmz = CNormalizerMinMax()
        self.tr.X /= 255
        self.ts.X /= 255

        transform = transforms.Lambda(lambda x: x.reshape(-1, 1, 28, 28))

        self._tr_loader = CDataLoaderPyTorch(self.tr.X, self.tr.Y,
                                             self.batch_size, shuffle=True,
                                             transform=transform).get_loader()

        self._ts_loader = CDataLoaderPyTorch(self.ts.X, self.ts.Y,
                                             self.batch_size, shuffle=False,
                                             transform=transform).get_loader()

    def _test_dtypes(self):
        self.logger.info("\tTest data types")
        assert(isinstance(self.tr, CDataset))
        assert(isinstance(self.tr.X[0, :], CArray))
        assert(isinstance(self.tr.Y[0, :], CArray))

    def _test_shapes(self, x_shape):
        """
        Test the shape of the loaded datasets.

        Parameters
        ----------
        x_shape : tuple
            shape of the expected input shape
        """
        self.logger.info("\tTest shapes")
        # test number of batches
        assert (len(self._tr_loader) == ceil(self.n_samples_tr / self.batch_size))
        assert (len(self._ts_loader) == ceil(self.n_samples_ts / self.batch_size))

        # test number of samples
        assert (len(self._tr_loader.dataset) == self.n_samples_tr)
        assert (len(self._ts_loader.dataset) == self.n_samples_ts)

        # test size of the samples
        x, y = next(iter(self._tr_loader))
        assert (x.shape == x_shape)
        assert (y.shape == (self.batch_size,))

    def test_blobs(self):
        self.logger.info("______________________________________")
        self.logger.info("Test blob dataset")
        self.logger.info("______________________________________")
        self._dataset_creation_blobs()
        self._test_dtypes()
        self._test_shapes(x_shape=(self.batch_size, 1, self.n_features))

    def test_mnist(self):
        self.logger.info("______________________________________")
        self.logger.info("Test MNIST dataset")
        self.logger.info("______________________________________")
        self._dataset_creation_mnist()
        self._test_dtypes()
        self._test_shapes(x_shape=(self.batch_size, 1, 1, 28, 28))

