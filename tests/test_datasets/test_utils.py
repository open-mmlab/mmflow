# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from tempfile import TemporaryDirectory

import mmcv
import numpy as np
import pytest
from PIL import Image

from mmflow.datasets.utils import (adjust_gamma, adjust_hue, flow_from_bytes,
                                   visualize_flow, write_flow)

img1_ = '../data/0000000-img_0.png'


@pytest.mark.parametrize('hue_factor', [0., 0.25, -0.25])
def test_adjust_hue(hue_factor):
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), img1_), channel_order='bgr')
    H, W, _ = img.shape

    # img = img[:H // 40, :W // 40, :]
    with pytest.raises(AssertionError):
        adjust_hue(img, 1.)

    img_hue = adjust_hue(img, hue_factor=hue_factor)

    import torchvision.transforms.functional as F

    img_PIL = Image.open(osp.join(osp.dirname(__file__), img1_))

    # PIL Image default mode is RGB
    assert np.allclose(img[:, :, ::-1], np.array(img_PIL), atol=1, rtol=1)

    img_hue_rgb_target = np.array(F.adjust_hue(img_PIL, hue_factor=hue_factor))

    assert np.allclose(
        img_hue[:, :, ::-1], img_hue_rgb_target, atol=2., rtol=2.)


@pytest.mark.parametrize('gamma',
                         [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
def test_adjust_gamma(gamma):

    img = mmcv.imread(osp.join(osp.dirname(__file__), img1_))

    with pytest.raises(AssertionError):
        adjust_gamma(img, -1.)

    with pytest.raises(AssertionError):
        img_normalized = img / 255.
        adjust_gamma(img_normalized, 1.0)

    img_gamma = adjust_gamma(img, gamma=gamma)

    img_PIL = Image.open(osp.join(osp.dirname(__file__), img1_))

    import torchvision.transforms.functional as F

    img_gamma_target = np.array(F.adjust_gamma(img_PIL, gamma=gamma))

    assert np.allclose(img_gamma[:, :, ::-1], img_gamma_target, atol=1, rtol=1)


def test_visualize_flow():

    height = 151
    width = 151
    flow = np.zeros((height, width, 2), dtype=np.float32)
    with TemporaryDirectory() as tmpdirname:
        visualize_flow(flow, save_file=osp.join(tmpdirname, 'flow.png'))
        write_flow(flow, osp.join(tmpdirname, 'flow.flo'))


def test_flow_from_bytes():
    filename = '../data/0000000-flow_01.flo'
    file_client = mmcv.FileClient(backend='disk')
    flow_bytes = file_client.get(osp.join(osp.dirname(__file__), filename))
    flow = flow_from_bytes(flow_bytes, filename[-3:])
    assert flow.shape[-1] == 2 and len(flow.shape) == 3
