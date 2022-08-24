# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Dict, Optional, Sequence, Union

import torch
from mmengine.model import BaseDataPreprocessor

from mmflow.registry import MODELS


@MODELS.register_module()
class FlowDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for optical flow tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 sigma_range: Optional[Sequence] = None,
                 clamp_range: Optional[Sequence] = None,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False) -> None:
        super().__init__()
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                '`mean` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean)} values')
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std)} values')
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        if sigma_range is not None:
            self.register_buffer('sigma_range', torch.tensor(sigma_range),
                                 False)
            self.register_buffer('clamp_range', torch.tensor(clamp_range),
                                 False)

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """

        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        img1s = [input_[0, ...] for input_ in inputs]
        img2s = [input_[1, ...] for input_ in inputs]

        if self.channel_conversion and img1s[0].size(0) == 3:
            img1s = [_img1[[2, 1, 0], ...] for _img1 in img1s]
            img2s = [_img2[[2, 1, 0], ...] for _img2 in img2s]

        img1s = [_img1.float() for _img1 in img1s]
        img2s = [_img2.float() for _img2 in img2s]

        if self._enable_normalize:
            img1s = [(img1 - self.mean) / self.std for img1 in img1s]
            img2s = [(img2 - self.mean) / self.std for img2 in img2s]

        if training and hasattr(self, 'sigma_range'):
            # Add Noise
            for i in range(len(img1s)):
                # create new sigma for each image pair
                sigma = torch.tensor(
                    random.uniform(*self.sigma_range), device=self.device)
                img1s[i] = torch.clamp(
                    img1s[i] + torch.randn_like(img1s[i]) * sigma,
                    min=self.clamp_range[0],
                    max=self.clamp_range[1])
                img2s[i] = torch.clamp(
                    img2s[i] + torch.randn_like(img2s[i]) * sigma,
                    min=self.clamp_range[0],
                    max=self.clamp_range[1])

        img1s = torch.stack(img1s, dim=0)
        img2s = torch.stack(img2s, dim=0)

        return dict(
            inputs=torch.cat((img1s, img2s), dim=1), data_samples=data_samples)
