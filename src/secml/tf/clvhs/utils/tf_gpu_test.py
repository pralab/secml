import tensorflow as tf
import logging


if __name__ == '__main__':

    if tf.test.gpu_device_name():
        logging.info('Default GPU Device: {}'.format(
            tf.test.gpu_device_name()))
    else:
        logging.warn(
            "The GPU is not used by tensorflow. "
            "We suggest you installing the GPU version of TF")
