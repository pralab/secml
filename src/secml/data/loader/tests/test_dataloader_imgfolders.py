from secml.testing import CUnitTest

from secml.data.loader import CDataLoaderImgFolders
from secml.array import CArray
from secml.utils import fm


class TestCDataLoaderImgFolders(CUnitTest):
    """Unit test for CDataLoaderImgFolders."""

    def setUp(self):

        self.dataloader = CDataLoaderImgFolders()

    def test_load_img(self):
        """Testing img dataset loading."""

        dl = CDataLoaderImgFolders()

        self.logger.info("Testing loading rgb dataset...")

        ds_rgb_path = fm.join(fm.abspath(__file__), "ds_rgb")

        ds = dl.load(ds_path=ds_rgb_path, img_format='jpeg')

        self.logger.info(
            "Loaded {:} images of {:} features, {:} classes".format(
                ds.num_samples, ds.num_features, ds.num_classes))

        self.assertEqual((2, 151875), ds.X.shape)
        self.assertEqual(2, ds.num_classes)
        self.assertTrue((ds.header.img_w == 225).all())
        self.assertTrue((ds.header.img_h == 225).all())
        self.assertTrue((ds.header.img_c == 3).all())

        self.logger.info("Testing loading grayscale dataset...")

        ds_gray_path = fm.join(fm.abspath(__file__), "ds_gray")

        ds = dl.load(ds_path=ds_gray_path, img_format='jpeg')

        self.logger.info(
            "Loaded {:} images of {:} features, {:} classes".format(
                ds.num_samples, ds.num_features, ds.num_classes))

        self.assertEqual((2, 50625), ds.X.shape)
        self.assertEqual(2, ds.num_classes)
        self.assertTrue((ds.header.img_w == 225).all())
        self.assertTrue((ds.header.img_h == 225).all())
        self.assertTrue((ds.header.img_c == 1).all())

    def test_load_paths(self):
        """Testing img dataset path loading."""
        dl = CDataLoaderImgFolders()

        self.logger.info("Testing loading paths of rgb dataset...")

        ds_rgb_path = fm.join(fm.abspath(__file__), "ds_rgb")

        ds = dl.load(ds_path=ds_rgb_path, img_format='jpeg', load_data=False)

        self.logger.info(
            "Loaded {:} images of {:} features, {:} classes".format(
                ds.num_samples, ds.num_features, ds.num_classes))

        # TODO: USE 'U' AFTER TRANSITION TO PYTHON 3
        self.assertIn(ds.X.dtype.char, ('S', 'U'))

        # Checking behavior of `get_labels_ovr`
        ovr = ds.get_labels_ovr(pos_label='tiger')  # Y : ['coyote', 'tiger']
        self.assert_array_equal(ovr, CArray([0, 1]))


if __name__ == '__main__':
    CUnitTest.main()
