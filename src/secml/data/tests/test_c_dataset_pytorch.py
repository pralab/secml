from secml.testing import CUnitTest

try:
    import torch
except ImportError:
    CUnitTest.importskip("torch")
else:
    import torch

from secml.data.c_dataset_pytorch import CDatasetPyTorch
from secml.data.loader import CDLRandom


class TestCDatasetPyTorch(CUnitTest):
    """Unittests for CDatasetPyTorch."""

    def setUp(self):
        self.n_classes = 3
        self.n_features = 5


        self.ds = CDLRandom(n_classes=self.n_classes,
                            n_features=self.n_features,
                            n_informative=self.n_features,
                            n_redundant=0).load()
        self.logger.info("num_samples: {}, num_classes: {:}".format(
            self.ds.num_samples, self.ds.num_classes))

    def test_convert(self):
        """Test converting a CDataset into a CDatasetPyTorch."""
        torch_ds = CDatasetPyTorch(data=self.ds.X, labels=self.ds.Y)
        self.assertEqual(torch_ds.X.ndim, 2)
        self.assertEqual(torch_ds.Y.ndim, 1)
        self.assertEqual(len(torch_ds), self.ds.num_samples)

        # Now we try setting only the samples
        torch_ds = CDatasetPyTorch(self.ds.X)
        self.assertEqual(torch_ds.X.ndim, 2)
        self.assertEqual(torch_ds.Y, None)
        self.assertEqual(len(torch_ds), self.ds.num_samples)

        # Now we try setting samples and labels separately
        torch_ds = CDatasetPyTorch(self.ds.X, self.ds.Y)
        self.assertEqual(torch_ds.X.ndim, 2)
        self.assertEqual(torch_ds.Y.ndim, 1)
        self.assertEqual(len(torch_ds), self.ds.num_samples)

        # Now we try setting samples and labels (OVA) separately
        torch_ds = CDatasetPyTorch(self.ds.X, self.ds.get_labels_onehot())
        self.assertEqual(torch_ds.X.ndim, 2)
        self.assertEqual(torch_ds.Y.ndim, 2)
        self.assertEqual(len(torch_ds), self.ds.num_samples)

        # Passing a ds and setting labels should raise TypeError
        with self.assertRaises(TypeError):
            CDatasetPyTorch(self.ds, labels=self.ds.Y)

    def test_getitem(self):
        """Test getitem from CDatasetPyTorch."""
        torch_ds = CDatasetPyTorch(self.ds)

        i = 10  # Index for getitem
        res = torch_ds[i]
        self.logger.info("torch_ds[{:}]:\n{:}".format(i, res))

        # Expanding
        samples, labels = res

        self.assertEqual(type(samples), torch.Tensor)
        self.assertEqual(type(labels), torch.Tensor)

        self.assertEqual(samples.shape, (1, self.ds.num_features))
        self.assertEqual(labels.shape, ())  # 0-D

        # Testing again with 2-D labels
        torch_ds = CDatasetPyTorch(self.ds.X, self.ds.get_labels_onehot())

        i = 10  # Index for getitem
        res = torch_ds[i]
        self.logger.info("torch_ds[{:}]:\n{:}".format(i, res))

        # Expanding
        samples, labels = res

        self.assertEqual(type(samples), torch.Tensor)
        self.assertEqual(type(labels), torch.Tensor)

        self.assertEqual(samples.shape, (1, self.ds.num_features))
        self.assertEqual(labels.shape, (1, self.ds.num_classes))

        with self.assertRaises(ValueError):
            # Only integer indexing is supported
            torch_ds[[2, 3]]


if __name__ == '__main__':
    CUnitTest.main()
