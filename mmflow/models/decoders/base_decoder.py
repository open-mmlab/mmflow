# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import List, Optional, Sequence, Union

import torch.nn.functional as F
from mmengine.data import PixelData
from mmengine.model import BaseModule
from torch import Tensor

from mmflow.data import FlowDataSample
from mmflow.utils import SampleList, TensorDict


class BaseDecoder(BaseModule):
    """Base class for decoder.

    Args:
        init_cfg (dict, list, optional): Config dict of weights initialization.
            Default: None.
    """

    def __init__(self, init_cfg: Optional[Union[dict, list]] = None) -> None:

        super().__init__(init_cfg=init_cfg)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Placeholder of forward function."""
        pass

    @abstractmethod
    def loss(self, *args, **kwargs):
        """Placeholder of forward function when model training."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Placeholder of forward function when model testing."""
        pass

    @abstractmethod
    def loss_by_feat(self, *args, **kwargs) -> TensorDict:
        """Placeholder for model computing losses."""
        pass

    def postprocess_result(
            self, results: Sequence[TensorDict],
            batch_img_metas: Sequence[dict]) -> Sequence[FlowDataSample]:
        """Reverted flow as original size of ground truth.

        Args:
            flow_result (Sequence[Dict[str, Tensor]]): predicted results of
                optical flow.
            batch_img_metas (Sequence[dict]): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[FlowDataSample]: the reverted predicted optical flow.
        """
        assert len(results) == len(batch_img_metas)

        data_samples = []
        for result, img_meta in zip(results, batch_img_metas):
            ori_H, ori_W = img_meta['ori_shape']
            pad = img_meta.get('pad', None)
            w_scale, h_scale = img_meta.get('scale_factor', (None, None))
            data_sample = FlowDataSample(**{'metainfo': img_meta})
            for key, f in result.items():
                if f is not None:
                    # shape is 2, H, W
                    H, W = f.shape[1:]
                    if pad is not None:
                        f = f[:, pad[0][0]:(H - pad[0][1]),
                              pad[1][0]:(W - pad[1][1])]

                    elif (w_scale is not None and h_scale is not None):
                        f = F.interpolate(
                            f[None],
                            size=(ori_H, ori_W),
                            mode='bilinear',
                            align_corners=False).squeeze(0)
                        f[0, :, :] = f[0, :, :] / w_scale
                        f[1, :, :] = f[1, :, :] / h_scale
                flow_data = PixelData(**{'data': f})
                data_sample.set_data({'pred_' + key: flow_data})

            data_samples.append(data_sample)

        return data_samples

    def predict_by_feat(self, flow_results: Tensor,
                        batch_img_metas: List[dict]) -> SampleList:
        """Predict list of obj:`FlowDataSample` from flow tensor.

        Args:
            flow_results (Tensor): Input flow tensor.
            batch_img_metas (Sequence[dict]): meta data of image to revert
                the flow to original ground truth size. Defaults to None.


        Returns:
            Sequence[FlowDataSample]: the reverted predicted optical flow.
        """
        H, W = batch_img_metas[0]['img_shape'][:2]
        # resize flow to the size of images after augmentation.
        flow_results = F.interpolate(
            flow_results, size=(H, W), mode='bilinear', align_corners=False)

        flow_results = flow_results * self.flow_div

        # unravel batch dim,
        flow_results = list(flow_results)
        results = [dict(flow_fw=f) for f in flow_results]

        return self.postprocess_result(
            results, batch_img_metas=batch_img_metas)
