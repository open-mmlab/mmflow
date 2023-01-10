# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    Args:

        img (ndarray): Image to be adjust with BGR order.

        value (float): the amount of shift in H channel and must be in the
            interval [-0.5, 0.5].. 0.5 and -0.5 give complete reversal of hue
            channel in HSV space in positive and negative direction
            respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will
            give an image with complementary colors while 0 gives the original
            image.

    Returns:
        ndarray: The hue-adjusted image.
    """
    if hue_factor is None:
        return img
    else:
        assert hue_factor >= -0.5 and hue_factor <= 0.5
        img = mmcv.bgr2hsv(img)
        img[:, :, 0] = (img[:, :, 0].astype(np.int_) + int(hue_factor * 180.) +
                        180) % 180
        img = mmcv.hsv2bgr(img)
        return img


def adjust_gamma(img, gamma=1.0):
    """Using gamma correction to process the image.

    Args:
        img (ndarray): Image to be adjusted. uint8 datatype.

        gamma (float or int): Gamma value used in gamma correction. gamma is a
            positive value. Note: gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter. Default: 1.0.
    """

    assert isinstance(gamma, float) or isinstance(gamma, int)
    assert gamma > 0

    assert img.dtype == 'uint8'

    table = ((np.arange(256) / 255.) ** gamma * (255 + 1 - 1e-3))\
        .astype('uint8')

    adjusted_img = mmcv.lut_transform(np.array(img, dtype=np.uint8), table)

    return adjusted_img
