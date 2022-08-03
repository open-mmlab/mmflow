# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from math import ceil
from typing import Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.image import adjust_brightness, adjust_color, adjust_contrast
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from numpy import random

from mmflow.registry import TRANSFORMS
from ..utils import adjust_gamma, adjust_hue

img_keys = ['img1', 'img2']

flow_keys = ['gt_flow_fw', 'gt_flow_bw']

occ_keys = ['gt_occ_fw', 'gt_occ_bw']

valid_keys = ['gt_valid_fw', 'gt_valid_bw']


@TRANSFORMS.register_module()
class SpacialTransform(BaseTransform):
    """Spacial Transform API for RAFT.

    Required Keys:

    - img1
    - img2
    - gt_flow_fw (optional)
    - gt_flow_bw (optional)
    - gt_occ_fw (optional)
    - gt_occ_bw (optional)
    - gt_valid_fw (optional)
    - gt_valid_bw (optional)


    Modified Keys:

    - img1
    - img2
    - gt_flow_fw (optional)
    - gt_flow_bw (optional)
    - gt_occ_fw (optional)
    - gt_occ_bw (optional)
    - gt_valid_fw (optional)
    - gt_valid_bw (optional)


    Added Keys:

    - scale

    Args:
        spacial_prob (float): probability to do spacial transform.
        stretch_prob (float): probability to do stretch.
        crop_size (tuple, list): the base size for resize.
        min_scale (float): the exponent for min scale. Defaults to -0.2.
        max_scale (float): the exponent for max scale. Defaults to 0.5.
    Returns:
        dict: Resized results, 'img_shape',
    """

    def __init__(self,
                 spacial_prob: float,
                 stretch_prob: float,
                 crop_size: Sequence,
                 min_scale: float = -0.2,
                 max_scale: float = 0.5,
                 max_stretch: float = 0.2) -> None:
        super().__init__()
        assert spacial_prob >= 0. and spacial_prob <= 1. and isinstance(
            spacial_prob, float)
        assert stretch_prob >= 0. and stretch_prob <= 1. and isinstance(
            stretch_prob, float)
        assert isinstance(
            crop_size, (tuple, list)) and len(crop_size) == 2 and isinstance(
                crop_size[0], int) and isinstance(crop_size[1], int)
        assert isinstance(min_scale, float)
        assert isinstance(max_scale, float)
        assert isinstance(max_stretch, float)
        self.spacial_prob = spacial_prob
        self.stretch_prob = stretch_prob
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_stretch = max_stretch

    def transform(self, results: dict) -> dict:
        """Call function to do spacial transform to images and annotation,
        including optical flow, occlusion mask and valid mask.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains transformed data and transform information.
        """

        if self._do_spacial_transform():

            H, W = results['img1'].shape[:2]
            self.newH, self.newW, self.x0, self.y0 = self._random_scale(H, W)

            if not results['sparse']:
                self._dense_flow_transform(results)
            else:
                self._sparse_flow_transform(results)
        else:
            results['scale'] = (1., 1.)

        return results

    @cache_randomness
    def _do_spacial_transform(self) -> bool:
        """Whether do spacial transform.

        Returns:
            bool: If True, images and flow will do spacial transform.
        """
        return np.random.rand() < self.spacial_prob

    @cache_randomness
    def _random_scale(self, H, W):
        """Sample image scale randomly.

        Args:
            H (int): The height of input images.
            W (int): The width of input images

        Returns:
            Tuple(int): The new height, width and coordinates for crop box.
        """
        min_scale = np.maximum((self.crop_size[0] + 8) / float(H),
                               (self.crop_size[1] + 8) / float(W))
        scale = 2**np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2**np.random.uniform(-self.max_stretch,
                                            self.max_stretch)
            scale_y *= 2**np.random.uniform(-self.max_stretch,
                                            self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)
        newW = int(W * float(scale_x) + 0.5)
        newH = int(H * float(scale_y) + 0.5)
        y0 = np.random.randint(0, newH - self.crop_size[0])
        x0 = np.random.randint(0, newW - self.crop_size[1])
        return newH, newW, x0, y0

    def _dense_flow_transform(self, results):
        """Transform for dense flow map.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.
        """
        # transform images
        for k in img_keys:
            if results.get(k) is not None:
                results[k], scale_x, scale_y, = self._resize_crop(results[k])
        results['scale'] = (scale_x, scale_y)
        results['img_shape'] = results['img1'].shape

        # transform flows
        for k in flow_keys:
            if results.get(k) is not None:
                flow, scale_x, scale_y = self._resize_crop(results[k])
                flow *= [scale_x, scale_y]
                results[k] = flow

        # transform occ
        for k in occ_keys:
            if results.get(k, None) is not None:
                results[k], _, _ = self._resize_crop(results[k])

        # transform valid
        for k in valid_keys:
            if results.get(k, None) is not None:
                results[k], _, _ = self._resize_crop(results[k])

    def _resize_crop(
        self,
        img: np.ndarray,
        interpolation: str = 'bilinear'
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """Spacial transform function.

        Args:
            img (ndarray): the images that will be transformed.

        Returns:
            Tuple[ndarray, float, float, int, int]: the transformed images,
                horizontal scale factor, vertical scale factor, coordinate of
                left-top point where the image maps will be crop from.
        """

        img_, scale_x, scale_y = mmcv.imresize(
            img, (self.newW, self.newH),
            return_scale=True,
            interpolation=interpolation)
        img_ = img_[self.y0:self.y0 + self.crop_size[0],
                    self.x0:self.x0 + self.crop_size[1]]

        return img_, scale_x, scale_y

    def _sparse_flow_transform(self, results):
        """Transform for sparse flow map.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.
        """
        # sparse spacial_transform for kitti/hd1k dataset
        # transform images
        for k in img_keys:
            if results.get(k) is not None:
                results[k], scale_x, scale_y, = self._resize_crop(results[k])
        results['scale'] = (scale_x, scale_y)
        results['img_shape'] = results['img1'].shape

        # transform flow_fw and valid
        flow, valid = self._sparse_flow_resize_crop(
            results['gt_flow_fw'],
            results['gt_valid_fw'],
            fx=scale_x,
            fy=scale_y)

        results['gt_flow_fw'] = flow
        results['gt_valid_fw'] = valid.astype(np.float32)

    def _sparse_flow_resize_crop(self,
                                 flow: np.ndarray,
                                 valid: np.ndarray,
                                 fx: float = 1.0,
                                 fy: float = 1.0) -> Sequence[np.ndarray]:
        """Resize sparse optical flow function.

        Args:
            flow (ndarray): optical flow data will be resized.
            valid (ndarray): valid mask for sparse optical flow.
            fx (float, optional): horizontal scale factor. Defaults to 1.0.
            fy (float, optional): vertical scale factor. Defaults to 1.0.

        Returns:
            Sequence[ndarray]: the transformed flow map and valid mask.
        """
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1.
        flow_img = flow_img[self.y0:self.y0 + self.crop_size[0],
                            self.x0:self.x0 + self.crop_size[1]]
        valid_img = valid_img[self.y0:self.y0 + self.crop_size[0],
                              self.x0:self.x0 + self.crop_size[1]]
        return flow_img, valid_img

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(spacial_prob={self.spacial_prob} '
                     f'stretch_prob={self.stretch_prob} '
                     f'crop_size={self.crop_size} '
                     f'min_scale={self.min_scale} '
                     f'max_scale = {self.max_scale}')
        return repr_str


@TRANSFORMS.register_module()
class Validation(BaseTransform):
    """This Validation transform from RAFT is for return a mask for the flow is
    less than max_flow.

    Required Keys:

    - gt_flow_fw (optional)
    - gt_flow_bw (optional)
    - gt_valid_fw (optional)
    - gt_valid_bw (optional)


    Modified Keys:

    - gt_valid_fw (optional)
    - gt_valid_bw (optional)

    Args:
        max_flow (float, int): the max flow for validated flow.
    Returns:
        dict: Resized results, 'valid' and 'max_flow' keys are added into
            result dict.
    """

    def __init__(self, max_flow: Union[float, int]) -> None:
        assert isinstance(max_flow, (float, int))
        self.max_flow = max_flow

    def transform(self, results: dict) -> dict:
        """Call function to get the valid mask.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: dict added 'valid' key and its value.
        """

        for k in flow_keys:
            if results.get(k, None) is not None:
                flow = results[k]
                valid = ((np.abs(flow[:, :, 0]) < self.max_flow) &
                         (np.abs(flow[:, :, 1]) < self.max_flow))
                valid_key = k.replace('flow', 'valid')
                if results.get(valid_key, None) is not None:
                    results[valid_key] = (
                        valid.astype(np.float32) * results[valid_key])
                else:
                    results[valid_key] = valid.astype(np.float32)
        results['max_flow'] = self.max_flow
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_flow={self.img_scale})'
        return repr_str


@TRANSFORMS.register_module()
class Erase(BaseTransform):
    """Erase transform from RAFT is randomly erasing rectangular regions in
    img2 to simulate occlusions.

    Required Keys:

    - img2

    Modified Keys:

    - img2

    Added  Keys:

    - erase_num
    - erase_bounds

    Args:
        prob (float): the probability for erase transform.
        bounds (list, tuple): the bounds for erase regions (bound_x, bound_y).
        max_num (int): the max number of erase regions.

    Returns:
        dict: revised results, 'img2' and 'erase_num' are added into results.
    """

    def __init__(self,
                 prob: float,
                 bounds: Sequence = [50, 100],
                 max_num: int = 3) -> None:
        assert isinstance(prob, float), ('Probability for erase transform must'
                                         f' be float, but got {type(prob)}')
        assert prob >= 0 and prob <= 1, ('The range of probability is [0.,1.],'
                                         f' but got {type(prob)}')

        assert isinstance(max_num,
                          int), f'max_num must be int, but got {type(max_num)}'
        self.prob = prob
        self.bounds = bounds
        self.max_num = max_num

    @cache_randomness
    def _do_erase(self):
        """Whether do erase transform.

        Returns:
            bool: If True, do this transform.
        """
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """Call function to do erase on images.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: the values of 'img1' and 'img2' is updated, and add
                'erase_num' and 'erase_bounds' keys and their values.
        """
        img2 = results['img2']
        H, W, _ = img2.shape
        erase_bounds = []
        num = 0
        if self._do_erase():
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            num = np.random.randint(1, self.max_num)
            for _ in range(num):
                x0 = np.random.randint(0, W)
                y0 = np.random.randint(0, H)
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
                erase_bounds.append((y0, x0, y0 + dy, x0 + dx))
        results['img2'] = img2
        results['erase_num'] = num
        results['erase_bounds'] = erase_bounds

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(prob={self.prob} '
                     f'bounds={self.bounds} '
                     f'max_num={self.max_num})')

        return repr_str


@TRANSFORMS.register_module()
class InputResize(BaseTransform):
    """Resize images such that dimensions are divisible by 2^n.

    Required Keys:

    - img1
    - img2

    Modified Keys:

    - img1
    - img2
    - img_shape

    Added Keys:

    - scale_factor


    Args:
        exponent(int): the exponent n of 2^n

    Returns:
        dict: Resized results, 'img_shape', 'scale_factor' keys are added
            into result dict.
    """

    def __init__(self, exponent: int) -> None:
        super().__init__()
        assert isinstance(exponent, int)
        self.exponent = exponent

    def transform(self, results: dict) -> dict:
        """Call function to resize images and flow map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'scale_factor' keys are added
                into result dict.
        """
        self._resize_img(results)

        return results

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        img1 = results['img1']
        img2 = results['img2']
        times = int(2**self.exponent)
        H, W = img1.shape[:2]
        newH = int(ceil(H / times) * times)
        newW = int(ceil(W / times) * times)
        resized_img1 = mmcv.imresize(img1, (newW, newH), return_scale=False)
        resized_img2 = mmcv.imresize(img2, (newW, newH), return_scale=False)
        results['img1'] = resized_img1
        results['img2'] = resized_img2

        w_scale = newW / W
        h_scale = newH / H
        scale_factor = np.array([w_scale, h_scale], dtype=np.float32)
        results['scale_factor'] = scale_factor
        results['img_shape'] = resized_img1.shape

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(exponent={self.exponent})')
        return repr_str


@TRANSFORMS.register_module()
class InputPad(BaseTransform):
    """Pad images such that dimensions are divisible by 2^n used in test.

    Required Keys:

    - img1
    - img2

    Modified Keys:

    - img1
    - img2
    - img_shape

    Added Keys:

    - pad_shape
    - pad

    Args:
        exponent(int): the exponent n of 2^n
        mode(str): mode for numpy.pad(). Defaults to 'edge'.
        position(str) the position of origin image, and valid value is one of
            'center', 'left', 'right', 'top' and 'down'. Defaults to 'center'
    """

    def __init__(self,
                 exponent: int,
                 mode: str = 'edge',
                 position: str = 'center',
                 **kwargs) -> None:
        assert position in ('center', 'left', 'right', 'top', 'down')
        assert isinstance(exponent, int)
        self.exponent = exponent
        self.mode = mode
        self.position = position
        self.kwargs = kwargs

    def transform(self, results: dict) -> dict:
        """Call function to pad image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)

        return results

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.exponent``."""

        img1 = results['img1']
        img2 = results['img2']
        times = int(2**self.exponent)
        H, W = img1.shape[:2]
        pad_h = (((H // times) + 1) * times - H) % times
        pad_w = (((W // times) + 1) * times - W) % times
        if self.position == 'center':
            self._pad = [[pad_h // 2, pad_h - pad_h // 2],
                         [pad_w // 2, pad_w - pad_w // 2]]
        elif self.position == 'left':
            self._pad = [[pad_h // 2, pad_h - pad_h // 2], [0, pad_w]]
        elif self.position == 'right':
            self._pad = [[pad_h // 2, pad_h - pad_h // 2], [pad_w, 0]]
        elif self.position == 'top':
            self._pad = [[0, pad_h, pad_w // 2], [pad_w - pad_w // 2]]
        elif self.position == 'down':
            self._pad = [[pad_h, 0], [pad_w // 2, pad_w - pad_w // 2]]
        if len(img1.shape) > 2:
            self._pad.append([0, 0])
        padded_img1 = np.pad(img1, self._pad, mode=self.mode, **self.kwargs)
        padded_img2 = np.pad(img2, self._pad, mode=self.mode, **self.kwargs)
        results['img1'] = padded_img1
        results['img2'] = padded_img2
        results['pad_shape'] = padded_img1.shape
        results['img_shape'] = padded_img1.shape
        results['pad'] = self._pad[:2]

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(exponent={self.exponent} '
                     f'mode={self.mode} '
                     f'position={self.position})')
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Flip the image and flow map.

    Required Keys:

    - img1
    - img2
    - gt_flow_fw (optional)
    - gt_flow_bw (optional)
    - gt_occ_fw (optional)
    - gt_occ_bw (optional)
    - gt_valid_fw (optional)
    - gt_valid_bw (optional)


    Modified Keys:

    - img1
    - img2
    - gt_flow_fw (optional)
    - gt_flow_bw (optional)
    - gt_occ_fw (optional)
    - gt_occ_bw (optional)
    - gt_valid_fw (optional)
    - gt_valid_bw (optional)

    Added Keys:

    - flip
    - flip_direction

    Args:
        prob (float): The flipping probability.
        direction(str): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, prob, direction='horizontal'):
        assert isinstance(prob, (float, int)) and prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.prob = prob
        self.direction = direction

    @cache_randomness
    def do_flip(self):
        return np.random.rand() < self.prob

    def transform(self, results):
        """Call function to flip optical flow map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        if self.do_flip():
            self._flip(results)
            if 'flip' in results and 'flip_direction' in results:
                results['flip'].append(True)
                results['flip_direction'].append(self.direction)
            else:
                results['flip'] = [True]
                results['flip_direction'] = [self.direction]
        else:
            if 'flip' in results and 'flip_direction' in results:
                results['flip'].append(False)
                results['flip_direction'].append(None)
            else:
                results['flip'] = [False]
                results['flip_direction'] = [None]

        return results

    def _flip(self, results: dict) -> None:
        """Flip function for images and annotations.

        Args:
            results (dict): Result dict from loading pipeline.
        """

        # flip img
        for k in img_keys:
            results[k] = mmcv.imflip(results[k], direction=self.direction)

        # flip flow
        if self.direction == 'horizontal':
            coeff = [-1, 1]
        else:
            coeff = [1, -1]
        for k in flow_keys:
            if results.get(k, None) is not None:
                results[k] = mmcv.imflip(
                    results[k], direction=self.direction) * coeff
        # flip occ
        for k in occ_keys:
            if results.get(k, None) is not None:
                results[k] = mmcv.imflip(results[k], direction=self.direction)

        # flip valid mask
        for k in valid_keys:
            if results.get(k, None) is not None:
                results[k] = mmcv.imflip(
                    results[k], direction=self.direction).copy()

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(prob={self.prob}), '
        repr_str += f'(direction={self.direction})'
        return repr_str


@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    """Normalize the image.

    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        if results.get('channels_order') == 'RGB':
            self.to_rgb = False
            warnings.warn('The channels order is RBG, '
                          'and image will not convert it again')
        for k in img_keys:
            results[k] = mmcv.imnormalize(results[k], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        if self.to_rgb:
            results['channels_order'] = 'RGB'
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@TRANSFORMS.register_module()
class BGR2RGB(BaseTransform):
    """Convert image channels from BGR to RGB order.

    Returns:
        dict: results contained converted images.
    """

    def __init__(self):
        super().__init__()

    def transform(self, results):
        for k in img_keys:
            results[k] = mmcv.bgr2rgb(results[k])
        results['channels_order'] = 'RGB'
        return results


@TRANSFORMS.register_module()
class Rerange(BaseTransform):
    """Rerange the image pixel value.

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, results):
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """
        for k in img_keys:
            img = results[k]
            img_min_value = np.min(img)
            img_max_value = np.max(img)
            assert img_min_value < img_max_value
            # rerange to [0, 1]
            img = (img - img_min_value) / (img_max_value - img_min_value)
            # rerange to [min_value, max_value]
            img = img * (self.max_value - self.min_value) + self.min_value
            results[k] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(BaseTransform):
    """Random crop the image & flow.

    Required Keys:

    - img1
    - img2
    - gt_flow_fw (optional)
    - gt_flow_bw (optional)
    - gt_occ_fw (optional)
    - gt_occ_bw (optional)
    - gt_valid_fw (optional)
    - gt_valid_bw (optional)


    Modified Keys:

    - img1
    - img2
    - gt_flow_fw (optional)
    - gt_flow_bw (optional)
    - gt_occ_fw (optional)
    - gt_occ_bw (optional)
    - gt_valid_fw (optional)
    - gt_valid_bw (optional)
    - img_shape

    Added Keys:

    - crop_bbox

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    @cache_randomness
    def get_crop_bbox(self, img_shape: tuple) -> tuple:
        """Randomly get a crop bounding box.

        Args:
            img_shape (tuple): The shape of images

        Returns:
            tuple: The crop box for images cropping.s
        """
        margin_h = max(img_shape[0] - self.crop_size[0], 0)
        margin_w = max(img_shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results):
        """Call function to randomly crop images, flow maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        img_shape = copy.deepcopy(results['img_shape'])
        crop_bbox = self.get_crop_bbox(img_shape)

        # crop imgs
        for k in img_keys:
            if results.get(k, None) is not None:
                results[k] = self.crop(results[k], crop_bbox=crop_bbox)

        # crop flow
        for k in flow_keys:
            if results.get(k, None) is not None:
                results[k] = self.crop(results[k], crop_bbox=crop_bbox)

        # crop occ
        for k in occ_keys:
            if results.get(k, None) is not None:
                results[k] = self.crop(results[k], crop_bbox=crop_bbox)

        # crop valid
        for k in valid_keys:
            if results.get(k, None) is not None:
                results[k] = self.crop(results[k], crop_bbox=crop_bbox)

        results['img_shape'] = results['img1'].shape
        results['crop_bbox'] = crop_bbox

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast, saturation and hue of an
    image.

    Required Keys:

    - img1
    - img2

    Modified Keys:

    - img1
    - img2

    Args:
        asymmetric_prob (float): the probability to do color jitter for two
            images asymmetrically.
        brightness (float, tuple):  How much to jitter brightness.
            brightness_factor is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
            Should be non negative numbers.
        contrast (float, tuple):  How much to jitter contrast.
            contrast_factor is chosen uniformly from
            [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
            Should be non negative numbers.
        saturation (float, tuple):  How much to jitter saturation.
            saturation_factor is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
            Should be non negative numbers.
        hue (float, tuple): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given
            [min, max]. Should have 0<= hue <= 0.5 or
            -0.5 <= min <= max <= 0.5.
    """

    def __init__(self,
                 asymmetric_prob=0.,
                 brightness=0.,
                 contrast=0.,
                 saturation=0.,
                 hue=0.):
        assert isinstance(
            asymmetric_prob, float
        ), f'asymmetric_prob must be float, but got {type(asymmetric_prob)}'
        self.asymmetric_prob = asymmetric_prob

        self._brightness = self._check_input(brightness, 'brightness')
        self._contrast = self._check_input(contrast, 'contrast')
        self._saturation = self._check_input(saturation, 'saturation')
        self._hue = self._check_input(
            hue, 'hue', center=0., bound=(-0.5, 0.5), clip_first_on_zero=False)

    def _get_param(self):

        fn_idx = np.random.permutation(4)
        b = None if self._brightness is None else np.random.uniform(
            self._brightness[0], self._brightness[1])
        c = None if self._contrast is None else np.random.uniform(
            self._contrast[0], self._contrast[1])
        s = None if self._saturation is None else np.random.uniform(
            self._saturation[0], self._saturation[1])
        h = None if self._hue is None else np.random.uniform(
            self._hue[0], self._hue[1])

        return fn_idx, b, c, s, h

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, (float, int)):

            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f'{name} values should be between {bound}')

        elif isinstance(value, (tuple, list)) and len(value) == 2:

            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f'{name} values should be between {bound}')

        else:
            raise TypeError(
                f'{name} should be a single number or a list/tuple with '
                f'length 2, but got {value}.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def color_jitter(self, img):
        fn_idx, brightness, contrast, saturation, hue = self._get_param()

        img = img if isinstance(img, (list, tuple)) else [img]
        length = len(img)
        for i in fn_idx:
            if i == 0 and brightness:
                img = [adjust_brightness(i, brightness) for i in img]

            if i == 1 and contrast:
                img = [adjust_contrast(i, contrast) for i in img]

            if i == 2 and saturation:
                img = [adjust_color(i, saturation) for i in img]

            if i == 3 and hue:
                img = [adjust_hue(i, hue) for i in img]
        if length == 1:
            return img[0]
        return img

    @cache_randomness
    def _do_asymmetric(self) -> bool:
        """Random function for asymmetric.

        Returns:
            bool: Whether do color jitter for images asymmetrically.
        """

        return np.random.rand() < self.asymmetric_prob

    def transform(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        imgs = []
        for k in img_keys:
            imgs.append(results[k])

        # asymmetric
        if self._do_asymmetric():
            imgs_ = []
            for i in imgs:
                i = self.color_jitter(i)
                imgs_.append(i)
            imgs = imgs_
        else:
            # symmetric
            imgs = self.color_jitter(imgs)
        for i, k in enumerate(img_keys):
            results[k] = imgs[i]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'asymmetric_prob={self.asymmetric_prob}, '
                     f'brightness_range={self._brightness}, '
                     f'contrast_range={self._contrast}, '
                     f'saturation_range={self._saturation}, '
                     f'hue_range={self._hue}')
        return repr_str


@TRANSFORMS.register_module()
class PhotoMetricDistortion(BaseTransform):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5.

    The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            beta = random.uniform(-self.brightness_delta,
                                  self.brightness_delta)
            img_ = []
            for i_img in img:
                img_.append(self.convert(i_img, beta=beta))
            return img_
        else:
            return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img_ = []
            for i_img in img:
                img_.append(self.convert(i_img, alpha=alpha))
            return img_
        else:
            return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            alpha = random.uniform(self.saturation_lower,
                                   self.saturation_upper)
            img_ = []
            for i_img in img:

                i_img = mmcv.bgr2hsv(i_img)
                i_img[:, :, 1] = self.convert(i_img[:, :, 1], alpha=alpha)

                i_img = mmcv.hsv2bgr(i_img)
                img_.append(i_img)
            return img_
        else:
            return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            hue_val = random.randint(-self.hue_delta, self.hue_delta)
            img_ = []
            for i_img in img:

                i_img = mmcv.bgr2hsv(i_img)
                i_img[:, :, 0] = (i_img[:, :, 0].astype(int) + hue_val) % 180
                i_img = mmcv.hsv2bgr(i_img)
                img_.append(i_img)
            return img_
        else:
            return img

    def transform(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img = []
        for k in img_keys:
            img.append(results[k])
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        for i, k in enumerate(img_keys):
            results[k] = img[i]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@TRANSFORMS.register_module()
class GaussianNoise(BaseTransform):
    """Add Gaussian Noise to images.

    Add Gaussian Noise, with mean 0 and std sigma uniformly sampled from
    sigma_range, to images. And then clamp the images to clamp_range.

    Args:
        sigma_range(list(float) | tuple(float)): Uniformly sample sigma of
            gaussian noise in sigma_range. Default: (0, 0.04)
        clamp_range(list(float) | tuple(float)): The min and max value to clamp
            the images after adding gaussian noise.
            Default: (float('-inf'), float('inf')).
    """

    def __init__(self,
                 sigma_range=(0, 0.04),
                 clamp_range=(float('-inf'), float('inf'))):

        assert isinstance(sigma_range, (list, tuple))
        assert len(sigma_range) == 2
        assert 0 <= sigma_range[0] < sigma_range[1]

        self.sigma_range = sigma_range

        assert isinstance(clamp_range, (list, tuple))
        assert len(clamp_range) == 2
        assert clamp_range[0] < clamp_range[1]

        self.clamp_range = clamp_range

    def transform(self, results):
        """Call function to add gaussian noise to images. And then clamp the
        images to clamp_range.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        # create new meta 'sigma'
        results['sigma'] = random.uniform(*self.sigma_range)

        for k in img_keys:

            assert results[k].dtype == np.float32, \
                'Before add Gaussian noise, it needs do normalize.'

            results[k] += np.random.randn(
                *results['img_shape']) * results['sigma']

            results[k] = np.clip(
                results[k],
                a_min=self.clamp_range[0],
                a_max=self.clamp_range[1]).astype(np.float32)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(sigma_range={self.sigma_range})'


@TRANSFORMS.register_module()
class RandomGamma(BaseTransform):
    """Random gamma correction of images.

    Required Keys:

    - img1
    - img2

    Modified Keys:

    - img1
    - img2

    Added Keys:

    - gamma

    Note: gamma larger than 1 make the shadows darker, while gamma smaller than
    1 make dark regions lighter.

    Args:
        gamma_range(list | tuple): A list or tuple of length 2. Uniformly
            sample gamma from gamma_range. Defaults to (0.7, 1.5).
    """

    def __init__(self, gamma_range: Sequence = (0.7, 1.5)):

        assert isinstance(gamma_range, (list, tuple))

        assert len(gamma_range) == 2

        assert 0 <= gamma_range[0] <= gamma_range[1]

        self.gamma_range = gamma_range

    @cache_randomness
    def _random_gamma(self):
        return random.uniform(*self.gamma_range)

    def transform(self, results: dict) -> dict:
        """Call function to process images using gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        # create new meta 'gamma'
        results['gamma'] = self._random_gamma()

        for k in img_keys:
            results[k] = adjust_gamma(results[k], results['gamma'])

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(gamma_range={self.gamma_range})'
