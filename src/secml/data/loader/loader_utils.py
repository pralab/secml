"""
.. module:: LoaderUtils
   :synopsis: Collection of mixed utilities for Data Loaders

.. moduleauthor:: Marco Melis <marco.melis@unica.it>
.. moduleauthor:: Angelo Sotgiu

"""
from PIL import Image

__all__ = ['resize_img', 'crop_img']


def resize_img(img, shape):
    """Resize input image to desired shape.

    If the input image is bigger than desired, the LANCZOS filter
    will be used. If calculates the output pixel value using
    a truncated sinc filter on all pixels that may contribute
    to the output value.

    Otherwise, a LINEAR filter will be used. It calculates the output
    pixel value using linear interpolation on all pixels that may
    contribute to the output value.

    Parameters
    ----------
    img : PIL.Image.Image
        Image to be resized.
    shape : tuple
        Desired output image dimensions (height, width).

    Returns
    -------
    PIL.Image
        Resized image.


    """
    w, h = img.size

    if w != shape[1] or h != shape[0]:  # Resize only if necessary
        if w > shape[1] or h > shape[0]:
            # LANCZOS is a slow filter,
            # but has the best performance for downscaling
            interpolation = Image.LANCZOS
        else:  # LINEAR is faster and ok for upscaling
            interpolation = Image.LINEAR

        # Reverse the tuple as Pillow convention is (width, height)
        img = img.resize(shape[::-1], interpolation)

    return img


def crop_img(img, crop):
    """Extract a center crop of the input image.

    Parameters
    ----------
    img : PIL.Image.Image
        Image to be cropped.
    crop : tuple
        Dimensions of the desired crop (height, width).

    Returns
    -------
    PIL.Image
        Cropped image.

    Notes
    -----
    The image center will be computed by rounding the coordinates
    if necessary. Python round default behavior is toward the
    closest even decimal.

    """
    w, h = img.size

    if crop[1] >= w or crop[0] >= h:
        raise ValueError(
            "crop dimensions cannot be higher than {:}".format(img.size))

    x1 = int(round((w - crop[1]) / 2.))
    y1 = int(round((h - crop[0]) / 2.))

    return img.crop((x1, y1, x1 + crop[1], y1 + crop[0]))
