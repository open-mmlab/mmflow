# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import PixelData
from torch import Tensor

from mmflow.structures import FlowDataSample
from mmflow.utils import OptSampleList, SampleList, TensorDict


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
            self,
            flow_results: List[Dict],
            data_samples: OptSampleList = None) -> Sequence[FlowDataSample]:
        """Reverted flow as original size of ground truth.

        Args:
            flow_results (List[Dict]): List of predicted results.
            data_samples (list[:obj:`FlowDataSample`], optional): Each item
                contains the meta information of each image and corresponding
                annotations. Defaults to None.

        Returns:
            Sequence[FlowDataSample]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """

        only_prediction = False
        if data_samples is None:
            data_samples = []
            only_prediction = True
        else:
            assert len(flow_results) == len(data_samples)

        for i in range(len(flow_results)):
            if only_prediction:
                prediction = FlowDataSample()
                for key, f in flow_results[i].items():
                    prediction.set_data({
                        'pred_' + key:
                        PixelData(**{'data': flow_results[i]})
                    })
                data_samples.append(prediction)
            else:
                img_meta = data_samples[i].metainfo
                ori_H, ori_W = img_meta['ori_shape']
                pad = img_meta.get('pad', None)
                w_scale, h_scale = img_meta.get('scale_factor', (None, None))
                for key, f in flow_results[i].items():
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
                    data_samples[i].set_data(
                        {'pred_' + key: PixelData(**{'data': f})})
        return data_samples

    def predict_by_feat(self,
                        flow_results: Tensor,
                        data_samples: OptSampleList = None) -> SampleList:
        """Predict list of obj:`FlowDataSample` from flow tensor.

        Args:
            flow_results (Tensor): Predicted flow tensor.
            data_samples (list[:obj:`FlowDataSample`], optional): Each item
                contains the meta information of each image and corresponding
                annotations. Defaults to None.

        Returns:
            Sequence[FlowDataSample]: The batch of predicted optical flow
                with the same size of images before augmentation.
        """
        if data_samples is None:
            flow_results = flow_results * self.flow_div
            # unravel batch dim,
            flow_results = list(flow_results)
            flow_results = [dict(flow_fw=f) for f in flow_results]
            return self.postprocess_result(flow_results, data_samples=None)

        H, W = data_samples[0].metainfo['img_shape'][:2]
        # resize flow to the size of images after augmentation.
        flow_results = F.interpolate(
            flow_results, size=(H, W), mode='bilinear', align_corners=False)

        flow_results = flow_results * self.flow_div

        # unravel batch dim,
        flow_results = list(flow_results)
        flow_results = [dict(flow_fw=f) for f in flow_results]

        return self.postprocess_result(flow_results, data_samples=data_samples)
