# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.structures import BaseDataElement, PixelData


class FlowDataSample(BaseDataElement):
    """A data structure interface of MMFlow. They are used as interfaces
    between different components.

    The attributes in ``FlowDataSample`` are divided into several parts:

        - ``gt_flow_fw``(PixelData): The ground truth of optical flow from img1
            to img2.
        - ``gt_flow_bw``(PixelData): The ground truth of optical flow from img2
            to img1.
        - ``gt_occ_fw``(PixelData): Forward ground truth of occlusion pixel
        map.
        - ``gt_occ_bw``(PixelData): Backward ground truth of occlusion pixel
        map.
        - ``pred_flow_fw``(PixelData): The prediction of optical flow from img1
            to img2.
        - ``pred_flow_bw``(PixelData): The prediction of optical flow from img2
            to img1.
        - ``pred_occ_fw``(PixelData): Forward prediction of occlusion pixel
        map.
        - ``pred_occ_bw``(PixelData): Backward prediction of occlusion pixel
        map.
        - ``gt_valid_fw``(PixelData): The mask for valid pixel maps, which is
        used in dataset or is used for filtered flow by ``Validation`` in data
        transform.
        - ``gt_valid_bw``(PixelData): The mask for valid pixel maps, which is
        used in dataset or is used for filtered flow by ``Validation`` in data
        transform.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from mmengine.structures import PixelData
        >>> from mmflow.structures import FlowDataSample

        >>> data_sample = FlowDataSample()
        >>> img_meta = dict(img_shape=(3, 4, 3))
        >>> gt_flow_fw = PixelData(metainfo=img_meta)
        >>> gt_flow_fw.flow = torch.rand((3, 4, 2))
        >>> data_sample.gt_flow_fw = gt_flow_fw
        >>> print(data_sample)
        <FlowDataSample(

            META INFORMATION

            DATA FIELDS
            gt_flow_fw: <PixelData(

                    META INFORMATION
                    img_shape: (3, 4, 3)

                    DATA FIELDS
                    flow: tensor([[[0.8098, 0.4833],
                                [0.3728, 0.2867],
                                [0.2114, 0.5077],
                                [0.9779, 0.3627]],

                                [[0.9561, 0.6763],
                                [0.2590, 0.6944],
                                [0.7470, 0.1223],
                                [0.0703, 0.5286]],

                                [[0.6824, 0.9608],
                                [0.1438, 0.6182],
                                [0.4836, 0.2257],
                                [0.2003, 0.2354]]])
                ) at 0x7efd6cd88150>
            _gt_flow_fw: <PixelData(

                    META INFORMATION
                    img_shape: (3, 4, 3)

                    DATA FIELDS
                    flow: tensor([[[0.8098, 0.4833],
                                [0.3728, 0.2867],
                                [0.2114, 0.5077],
                                [0.9779, 0.3627]],

                                [[0.9561, 0.6763],
                                [0.2590, 0.6944],
                                [0.7470, 0.1223],
                                [0.0703, 0.5286]],

                                [[0.6824, 0.9608],
                                [0.1438, 0.6182],
                                [0.4836, 0.2257],
                                [0.2003, 0.2354]]])
                ) at 0x7efd6cd88150>
        ) at 0x7efd6ca023d0>
    """

    @property
    def gt_flow_fw(self) -> PixelData:
        return self._gt_flow_fw

    @gt_flow_fw.setter
    def gt_flow_fw(self, value: PixelData) -> None:
        self.set_field(value, '_gt_flow_fw', dtype=PixelData)

    @gt_flow_fw.deleter
    def gt_flow_fw(self) -> None:
        del self._gt_flow_fw

    @property
    def gt_flow_bw(self) -> PixelData:
        return self._gt_flow_bw

    @gt_flow_bw.setter
    def gt_flow_bw(self, value: PixelData) -> None:
        self.set_field(value, '_gt_flow_bw', dtype=PixelData)

    @gt_flow_bw.deleter
    def gt_flow_bw(self) -> None:
        del self._gt_flow_bw

    @property
    def gt_occ_fw(self) -> PixelData:
        return self._gt_occ_fw

    @gt_occ_fw.setter
    def gt_occ_fw(self, value: PixelData) -> None:
        self.set_field(value, '_gt_occ_fw', dtype=PixelData)

    @gt_occ_fw.deleter
    def gt_occ_fw(self) -> None:
        del self._gt_occ_fw

    @property
    def gt_occ_bw(self) -> PixelData:
        return self._gt_occ_bw

    @gt_occ_bw.setter
    def gt_occ_bw(self, value: PixelData) -> None:
        self.set_field(value, '_gt_occ_bw', dtype=PixelData)

    @gt_occ_bw.deleter
    def gt_occ_bw(self) -> None:
        del self._gt_occ_bw

    @property
    def pred_flow_fw(self) -> PixelData:
        return self._pred_flow_fw

    @pred_flow_fw.setter
    def pred_flow_fw(self, value: PixelData) -> None:
        self.set_field(value, '_pred_flow_fw', dtype=PixelData)

    @pred_flow_fw.deleter
    def pred_flow_fw(self) -> None:
        del self._pred_flow_fw

    @property
    def pred_flow_bw(self) -> PixelData:
        return self._pred_flow_bw

    @pred_flow_bw.setter
    def pred_flow_bw(self, value: PixelData) -> None:
        self.set_field(value, '_pred_flow_bw', dtype=PixelData)

    @pred_flow_bw.deleter
    def pred_flow_bw(self) -> None:
        del self._pred_flow_bw

    @property
    def pred_occ_fw(self) -> PixelData:
        return self._pred_occ_fw

    @pred_occ_fw.setter
    def pred_occ_fw(self, value: PixelData) -> None:
        self.set_field(value, '_pred_occ_fw', dtype=PixelData)

    @pred_occ_fw.deleter
    def pred_occ_fw(self) -> None:
        del self._pred_occ_fw

    @property
    def pred_occ_bw(self) -> PixelData:
        return self._pred_occ_bw

    @pred_occ_bw.setter
    def pred_occ_bw(self, value: PixelData) -> None:
        self.set_field(value, '_pred_occ_bw', dtype=PixelData)

    @pred_occ_bw.deleter
    def pred_occ_bw(self):
        del self._pred_occ_bw

    @property
    def gt_valid_fw(self) -> PixelData:
        return self._gt_valid_fw

    @gt_valid_fw.setter
    def gt_valid_fw(self, value: PixelData) -> None:
        self.set_field(value, '_gt_valid_fw', dtype=PixelData)

    @gt_valid_fw.deleter
    def gt_valid_fw(self) -> None:
        del self._gt_valid_fw

    @property
    def gt_valid_bw(self) -> PixelData:
        return self._gt_valid_bw

    @gt_valid_bw.setter
    def gt_valid_bw(self, value: PixelData) -> None:
        self.set_field(value, '_gt_valid_bw', dtype=PixelData)

    @gt_valid_bw.deleter
    def gt_valid_bw(self) -> None:
        del self._gt_valid_bw
