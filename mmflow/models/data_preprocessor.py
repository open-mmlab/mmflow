# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor

from mmflow.core import FlowDataSample
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

    def collate_data(
        self, data: Sequence[dict]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[list]]:
        """Collating and copying data to the target device.

        Collates the data sampled from dataloader into a list of tensor and
        list of labels, and then copies tensor to the target device.

        Subclasses could override it to be compatible with the custom format
        data sampled from custom dataloader.

        Args:
            data (Sequence[dict]): Data sampled from dataloader.

        Returns:
            Tuple[List[torch.Tensor], Optional[list]]: Unstacked list of input
            tensor and list of labels at target device.
        """
        # img1s is list of tensor with shape 3,H,W
        img1s = [
            data_['inputs'][0, ...].to(self._device).float() for data_ in data
        ]
        img2s = [
            data_['inputs'][1, ...].to(self._device).float() for data_ in data
        ]
        batch_data_samples: List[FlowDataSample] = []
        # Model can get predictions without any data samples.
        for _data in data:
            if 'data_sample' in _data:
                batch_data_samples.append(_data['data_sample'])
        # Move data from CPU to corresponding device.
        batch_data_samples = [
            data_sample.to(self._device) for data_sample in batch_data_samples
        ]

        if not batch_data_samples:
            batch_data_samples = None  # type: ignore

        return img1s, img2s, batch_data_samples

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """

        img1s, img2s, batch_data_samples = self.collate_data(data)

        if self.channel_conversion and img1s[0].size(0) == 3:
            img1s = [_img1[[2, 1, 0], ...] for _img1 in img1s]
            img2s = [_img2[[2, 1, 0], ...] for _img2 in img2s]

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

        return torch.cat((img1s, img2s), dim=1), batch_data_samples
