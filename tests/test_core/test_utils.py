# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmflow.core.utils import stack_batch


def test_stack_batch():
    # test non list input
    with pytest.raises(AssertionError):
        stack_batch(1, 1)
    # test invalid tensor dim
    with pytest.raises(AssertionError):
        img1_1 = torch.randn((3, 2, 2))
        img1_2 = torch.randn((1, 3, 2, 2))
        stack_batch([img1_1, img1_2], [img1_1, img1_2])
    with pytest.raises(AssertionError):
        img1 = torch.randn((1, 3, 2, 2))
        stack_batch([img1], [img1])
    # tes invalid channel
    with pytest.raises(AssertionError):
        img1_1 = torch.randn((3, 2, 2))
        img1_2 = torch.randn((1, 2, 2))
        stack_batch([img1_1, img1_2], [img1_1, img1_2])

    img1_1 = torch.randn((3, 2, 2))
    img1_2 = torch.randn((3, 2, 2))
    img2_1 = torch.randn((3, 2, 2))
    img2_2 = torch.randn((3, 2, 2))
    img1s, img2s = stack_batch([img1_1, img1_2], [img2_1, img2_2])
    assert img1s.shape == img2s.shape
