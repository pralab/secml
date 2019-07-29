from secml.testing import CUnitTest

from secml.data.loader import CDataLoaderImgClients
from secml.array import CArray
from secml.utils import fm


class TestCDataLoaderImgClients(CUnitTest):
    """Unit test for CDataLoaderImgClients."""

    def setUp(self):

        self.dataloader = CDataLoaderImgClients()

    def test_load_img(self):
        """Testing img dataset loading."""

        dl = CDataLoaderImgClients()

        self.logger.info("Testing loading clients dataset...")

        ds_path = fm.join(fm.abspath(__file__), "ds_clients")

        ds = dl.load(ds_path=ds_path, img_format='jpeg')

        self.logger.info(
            "Loaded {:} images of {:} features, {:} classes".format(
                ds.num_samples, ds.num_features, ds.num_classes))

        self.assertEqual((2, 151875), ds.X.shape)
        self.assertEqual(2, ds.num_classes)
        self.assertTrue((ds.header.img_w == 225).all())
        self.assertTrue((ds.header.img_h == 225).all())
        self.assertTrue((ds.header.img_c == 3).all())

    def test_load_paths(self):
        """Testing img dataset path loading."""
        dl = CDataLoaderImgClients()

        self.logger.info("Testing loading paths of clients dataset...")

        ds_path = fm.join(fm.abspath(__file__), "ds_clients")

        ds = dl.load(ds_path=ds_path, img_format='jpeg', load_data=False)

        self.logger.info(
            "Loaded {:} images of {:} features, {:} classes".format(
                ds.num_samples, ds.num_features, ds.num_classes))

        # TODO: USE 'U' AFTER TRANSITION TO PYTHON 3
        self.assertIn(ds.X.dtype.char, ('S', 'U'))

        # Checking correct label-img association
        self.assertEqual(ds.Y[0].item(),
                         fm.split(ds.X[0, :].item())[1].replace('.jpeg', ''))
        self.assertEqual(ds.Y[1].item(),
                         fm.split(ds.X[1, :].item())[1].replace('.jpeg', ''))

        # Checking behavior of `get_labels_ovr`
        ovr = ds.get_labels_ovr(pos_label='tiger')  # Y : ['coyote', 'tiger']
        self.assert_array_equal(ovr, CArray([0, 1]))


if __name__ == '__main__':
    CUnitTest.main()
