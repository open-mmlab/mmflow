# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.runner import BaseModule


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
    def forward_train(self, *args, **kwargs):
        """Placeholder of forward function when model training."""
        pass

    @abstractmethod
    def forward_test(self, *args, **kwargs):
        """Placeholder of forward function when model testing."""
        pass

    @abstractmethod
    def losses(self):
        """Placeholder for model computing losses."""
        pass

    def get_flow(
            self,
            flow_result: Sequence[Dict[str, np.ndarray]],
            img_metas: Sequence[dict] = None
    ) -> Sequence[Dict[str, np.ndarray]]:
        """Reverted flow as original size of ground truth.

        Args:
            flow_result (Sequence[Dict[str, ndarray]]): predicted results of
                optical flow.
            img_metas (Sequence[dict], optional): meta data of image to revert
                the flow to original ground truth size. Defaults to None.

        Returns:
            Sequence[Dict[str, ndarray]]: the reverted predicted optical flow.
        """
        if img_metas is not None:
            ori_shapes = [img_meta['ori_shape'] for img_meta in img_metas]
            img_shapes = [img_meta['img_shape'] for img_meta in img_metas]
            pad_shapes = [img_meta['pad_shape'] for img_meta in img_metas]

        if (img_metas is None
                or ori_shapes[0] == img_shapes[0] == pad_shapes[0]):
            return flow_result

        for i in range(len(flow_result)):

            pad = img_metas[i].get('pad', None)
            w_scale, h_scale = img_metas[i].get('scale_factor', (None, None))
            ori_shape = img_metas[i]['ori_shape']

            for key, f in flow_result[i].items():
                H, W = f.shape[:2]
                if pad is not None:
                    f = f[pad[0][0]:(H - pad[0][1]), pad[1][0]:(W - pad[1][1])]

                elif (w_scale is not None and h_scale is not None):
                    f = mmcv.imresize(
                        f,
                        (ori_shape[1], ori_shape[0]),  # size(w, h)
                        interpolation='bilinear',
                        return_scale=False)
                    f[:, :, 0] = f[:, :, 0] / w_scale
                    f[:, :, 1] = f[:, :, 1] / h_scale
                flow_result[i][key] = f

        return flow_result
