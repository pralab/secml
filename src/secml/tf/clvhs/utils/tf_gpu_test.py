from secml.utils import CUnitTest

import tensorflow as tf
import warnings


class TestGpuUsage(CUnitTest):
    """Test if the Gpu is used."""

    def test_GPU_usage(self):

        if tf.test.gpu_device_name():
            self.logger.info('Default GPU Device: {}'.format(
                tf.test.gpu_device_name()))
        else:
            warnings.warn("The GPU is not used by tensorflow. We suggest you "
                        "installing the GPU version of TF")


if __name__ == '__main__':
    CUnitTest.main()
