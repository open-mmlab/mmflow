# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from math import ceil
from typing import List, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
from mmcv.image import adjust_brightness, adjust_color, adjust_contrast
from numpy import random

from ..builder import PIPELINES
from ..utils import adjust_gamma, adjust_hue


def get_flow_keys(results: dict) -> List[str]:
    """Get keys of optical flow in results.

    Args:
        results (dict): The data that includes data and meta information and
            used in data augmentation pipeline.

    Returns:
        List: keys of optical flow in results.
    """
    flow_keys = []
    if 'ann_fields' in results:
        ann_keys = copy.deepcopy(results['ann_fields'])
        for k in ann_keys:
            if k.find('flow') > -1:
                flow_keys.append(k)
    return flow_keys


def get_img_keys(results: dict) -> List:
    """Get image keys in results.

    Args:
        results (dict): The data that includes data and meta information and
            used in data augmentation pipeline.

    Returns:
        List: Now it will return ['img1', 'img2'].
    """
    img_keys = copy.deepcopy(results['img_fields'])
    return img_keys


def get_map_keys(results: dict) -> list:
    """Get image and annotation keys in results.

    This annotation don't include 'valid' or 'valid_fw' and 'valid_bw'.

    Args:
        results (dict): The data that includes data and meta information and
            used in data augmentation pipeline.

    Returns:
        list: List of image keys and annotation keys.
    """
    img_fields = copy.deepcopy(results['img_fields'])
    if 'ann_fields' in results:
        return img_fields + results['ann_fields']
    else:
        return img_fields


def get_valid_keys(results: dict) -> list:
    """Get valid keys in results.

    Args:
        results (dict): The data that includes data and meta information and
            used in data augmentation pipeline.

    Returns:
        list: Now it will return ['valid'] or [] if there is not 'valid' in
            results.
    """
    flow_keys = get_flow_keys(results)
    if len(flow_keys) == 0:
        return []
    # -3 is for _gt
    valid_keys = [k.replace('flow', 'valid')[:-3] for k in flow_keys]

    if results.get(valid_keys[0], None) is None:
        return []
    else:
        return valid_keys


@PIPELINES.register_module()
class SpacialTransform:
    """Spacial Transform API for RAFT
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

    def __call__(self, results: dict) -> dict:
        """Call function to do spacial transform to images and annotation,
        including optical flow, occlusion mask and valid mask.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: The dict contains transformed data and transform information.
        """

        if np.random.rand() < self.spacial_prob:

            img_keys = get_img_keys(results)
            flow_keys = get_flow_keys(results)
            if results.get('valid') is None:
                map_keys = get_map_keys(results)
                flow_inds = [map_keys.index(k) for k in flow_keys]
                maps = [results[k] for k in map_keys]
                maps, scale_x, scale_y, _, _ = self.spacial_transform(maps)
                for idx in flow_inds:
                    maps[idx] *= [scale_x, scale_y]
                results['scale'] = (scale_x, scale_y)
                results['img_shape'] = maps[0].shape
                for i, k in enumerate(map_keys):
                    results[k] = maps[i]
            else:
                # sparse spacial_transform
                imgs = [results[k] for k in img_keys]
                imgs, scale_x, scale_y, x0, y0 = self.spacial_transform(imgs)
                for i, k in enumerate(img_keys):
                    results[k] = imgs[i]
                results['scale'] = (scale_x, scale_y)
                results['img_shape'] = imgs[0].shape
                flow, valid = self.resize_sparse_flow_map(
                    results['flow_gt'],
                    results['valid'],
                    fx=scale_x,
                    fy=scale_y,
                    x0=x0,
                    y0=y0)

                results['flow_gt'] = flow
                results['valid'] = valid.astype(np.float32)

        else:
            results['scale'] = (1., 1.)

        return results

    def resize_sparse_flow_map(self,
                               flow: np.ndarray,
                               valid: np.ndarray,
                               fx: float = 1.0,
                               fy: float = 1.0,
                               x0: int = 0,
                               y0: int = 0) -> Sequence[np.ndarray]:
        """Resize sparse optical flow function.

        Args:
            flow (ndarray): optical flow data will be resized.
            valid (ndarray): valid mask for sparse optical flow.
            fx (float, optional): horizontal scale factor. Defaults to 1.0.
            fy (float, optional): vertical scale factor. Defaults to 1.0.
            x0 (int, optional): abscissa of left-top point where the flow map
                will be crop from. Defaults to 0.
            y0 (int, optional): ordinate of left-top point where the flow map
                will be crop from. Defaults to 0.

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
        flow_img = flow_img[y0:y0 + self.crop_size[0],
                            x0:x0 + self.crop_size[1]]
        valid_img = valid_img[y0:y0 + self.crop_size[0],
                              x0:x0 + self.crop_size[1]]
        return flow_img, valid_img

    def spacial_transform(
            self,
            imgs: np.ndarray) -> Tuple[np.ndarray, float, float, int, int]:
        """Spacial transform function.

        Args:
            imgs (ndarray): the images that will be transformed.

        Returns:
            Tuple[ndarray, float, float, int, int]: the transformed images,
                horizontal scale factor, vertical scale factor, coordinate of
                left-top point where the image maps will be crop from.
        """
        H, W = imgs[0].shape[:2]
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

        newW, newH = int(W * float(scale_x) + 0.5), int(H * float(scale_y) +
                                                        0.5)

        y0 = np.random.randint(0, newH - self.crop_size[0])
        x0 = np.random.randint(0, newW - self.crop_size[1])

        imgs_ = []
        for img in imgs:
            img_, scale_x, scale_y = mmcv.imresize(
                img, (newW, newH), return_scale=True)
            img_ = img_[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            imgs_.append(img_)

        return imgs_, scale_x, scale_y, x0, y0

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(spacial_prob={self.spacial_prob} '
                     f'stretch_prob={self.stretch_prob} '
                     f'crop_size={self.crop_size} '
                     f'min_scale={self.min_scale} '
                     f'max_scale = {self.max_scale}')
        return repr_str


@PIPELINES.register_module()
class Validation:
    """This Validation transform from RAFT is for return a mask for the flow is
    less than max_flow.

    Args:
        max_flow (float, int): the max flow for validated flow.
    Returns:
        dict: Resized results, 'valid' and 'max_flow' keys are added into
            result dict.
    """

    def __init__(self, max_flow: Union[float, int]) -> None:
        assert isinstance(max_flow, (float, int))
        self.max_flow = max_flow

    def __call__(self, results: dict) -> dict:
        """Call function to get the valid mask.

        Args:
            results (dict): Result dict from :obj:`mmflow.BaseDataset`.

        Returns:
            dict: dict added 'valid' key and its value.
        """

        flow_keys = get_flow_keys(results)
        for k in flow_keys:
            flow = results[k]
            valid = ((np.abs(flow[:, :, 0]) < self.max_flow) &
                     (np.abs(flow[:, :, 1]) < self.max_flow))
            valid_key = k.replace('flow', 'valid')
            # [:-3] is for '_gt'.
            results[valid_key[:-3]] = valid.astype(np.float32)
        results['max_flow'] = self.max_flow
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_flow={self.img_scale})'
        return repr_str


@PIPELINES.register_module()
class Erase:
    """Erase transform from RAFT is randomly erasing rectangular regions in
    img2 to simulate occlusions.

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

    def __call__(self, results: dict) -> dict:
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
        if np.random.rand() < self.prob:
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


@PIPELINES.register_module()
class InputResize:
    """Resize images such that dimensions are divisible by 2^n
    Args:
        exponent(int): the exponent n of 2^n

    Returns:
        dict: Resized results, 'img_shape', 'scale_factor' keys are added
            into result dict.
    """

    def __init__(self, exponent) -> None:
        super().__init__()
        assert isinstance(exponent, int)
        self.exponent = exponent

    def __call__(self, results):
        """Call function to resize images and flow map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img_shape', 'scale_factor' keys are added
                into result dict.
        """
        img_keys = get_img_keys(results)
        imgs = [results[k] for k in img_keys]
        imgs, scale_factor = self._resize_img(imgs)

        for i, k in enumerate(img_keys):
            results[k] = imgs[i]
        results['scale_factor'] = scale_factor
        results['img_shape'] = imgs[0].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(exponent={self.exponent})')
        return repr_str

    def _resize_img(self, imgs):
        """Resize images with ``results['scale']``."""
        times = int(2**self.exponent)
        H, W = imgs[0].shape[:2]
        newH = int(ceil(H / times) * times)
        newW = int(ceil(W / times) * times)
        imgs_resize = []
        for img in imgs:
            img_ = mmcv.imresize(img, (newW, newH), return_scale=False)
            imgs_resize.append(img_)
        w_scale = newW / W
        h_scale = newH / H

        scale_factor = np.array([w_scale, h_scale], dtype=np.float32)
        return imgs_resize, scale_factor


@PIPELINES.register_module()
class InputPad:
    """Pad images such that dimensions are divisible by 2^n used in test.

    Args:
        exponent(int): the exponent n of 2^n
        mode(str): mode for numpy.pad(). Defaults to 'edge'.
        position(str) the position of origin image, and valid value is one of
            'center', 'left', 'right', 'top' and 'down'. Defaults to 'center'
    """

    def __init__(self, exponent, mode='edge', position='center', **kwargs):
        assert position in ('center', 'left', 'right', 'top', 'down')
        assert isinstance(exponent, int)
        self.exponent = exponent
        self.mode = mode
        self.position = position
        self.kwargs = kwargs

    def __call__(self, results):
        img_keys = get_img_keys(results)
        imgs = [results[k] for k in img_keys]
        imgs = self.pad(imgs)
        for i, k in enumerate(img_keys):
            results[k] = imgs[i]
        results['pad_shape'] = imgs[0].shape
        results['pad'] = self._pad[:2]

        return results

    def pad(self, imgs):

        times = int(2**self.exponent)
        H, W = imgs[0].shape[:2]
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
        if len(imgs[0].shape) > 2:
            self._pad.append([0, 0])
        imgs = [
            np.pad(img, self._pad, mode=self.mode, **self.kwargs)
            for img in imgs
        ]
        return imgs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(exponent={self.exponent} '
                     f'mode={self.mode} '
                     f'position={self.position})')
        return repr_str


@PIPELINES.register_module()
class RandomFlip:
    """Flip the image and flow map.

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

    def __call__(self, results):
        """Call function to flip optical flow map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.prob else False
        if flip:
            # flip image
            map_keys = get_map_keys(results)
            flow_keys = get_flow_keys(results)
            valid_keys = get_valid_keys(results)

            for k in map_keys:
                results[k] = mmcv.imflip(results[k], direction=self.direction)

            for valid_key in valid_keys:
                results[valid_key] = mmcv.imflip(
                    results[valid_key], direction=self.direction).copy()

            # flip flow
            if self.direction == 'horizontal':
                coeff = [-1, 1]
            else:
                coeff = [1, -1]
            for fk in flow_keys:
                results[fk] = results[fk] * coeff

        if 'flip' in results and 'flip_direction' in results:
            results['flip'].append(flip)
            results['flip_direction'].append(self.direction)
        else:
            results['flip'] = [flip]
            results['flip_direction'] = [self.direction]

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class Normalize:
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

    def __call__(self, results):
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
        img_keys = get_img_keys(results)
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


@PIPELINES.register_module()
class BGR2RGB:
    """Convert image channels from BGR to RGB order.

    Returns:
        dict: results contained converted images.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, results):
        img_keys = get_img_keys(results)
        for k in img_keys:
            results[k] = mmcv.bgr2rgb(results[k])
        results['channels_order'] = 'RGB'
        return results


@PIPELINES.register_module()
class Rerange:
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

    def __call__(self, results):
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """
        img_keys = get_img_keys(results)
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


@PIPELINES.register_module()
class RandomCrop:
    """Random crop the image & flow.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def get_crop_bbox(self, img_shape):
        """Randomly get a crop bounding box."""
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

    def __call__(self, results):
        """Call function to randomly crop images, flow maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        map_keys = get_map_keys(results)
        img_shape = copy.deepcopy(results['img_shape'])
        valid_keys = get_valid_keys(results)
        crop_bbox = self.get_crop_bbox(img_shape)

        for k in map_keys:
            results[k] = self.crop(results[k], crop_bbox=crop_bbox)

        for k in valid_keys:
            results[k] = self.crop(results[k], crop_bbox=crop_bbox)
        results['img_shape'] = results['img1'].shape
        results['crop_bbox'] = crop_bbox

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class ColorJitter:
    """Randomly change the brightness, contrast, saturation and hue of
    an image.
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

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img_keys = get_img_keys(results)
        imgs = []
        for k in img_keys:
            imgs.append(results[k])
        asym = np.random.rand()
        # asymmetric
        if asym < self.asymmetric_prob:
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


@PIPELINES.register_module()
class PhotoMetricDistortion:
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

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        img_keys = get_img_keys(results)
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


@PIPELINES.register_module()
class RandomRotation:
    """Random rotation of the image from -angle to angle (in degrees).

    .. note: This augmentation is for dense optical flow data, not for sparse
    optical flow data.

    Args:
        prob (float): The rotation probability.
        angle (float): max angle of the rotation in the range from -180 to 180.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self, prob, angle, auto_bound=False):
        assert isinstance(prob, (float)) and prob >= 0. and prob <= 1.
        assert isinstance(angle, float) and angle >= -180. and angle <= 180.
        self.prob = prob
        self.angle = angle
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate the images and optical flow map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Processed results.
        """
        rotate = True if np.random.rand() < self.prob else False
        angle = random.uniform(-self.angle, self.angle)
        if rotate:
            angle_rad = angle * np.pi / 180.
            cos = np.cos(angle_rad)
            sin = np.sin(angle_rad)

            map_keys = get_map_keys(results)
            flow_keys = get_flow_keys(results)
            for k in map_keys:
                img = results[k]
                img = mmcv.imrotate(img, angle, auto_bound=self.auto_bound)
                results[k] = img

            # Rotation matrix in image coordinate (origin is assumed to be
            # the top-left corners) with the angle that positive values mean
            # clockwise rotation
            # |cos  -sin|
            # |sin  cos|
            for fk in flow_keys:
                flow_ = copy.deepcopy(results[fk])
                results[fk][:, :, 0] = \
                    cos * flow_[:, :, 0] - sin * flow_[:, :, 1]
                results[fk][:, :, 1] = \
                    sin * flow_[:, :, 0] + cos * flow_[:, :, 1]
        results['rotate'] = rotate
        results['rotate_angle'] = angle
        if self.auto_bound:
            results['img_shape'] = results['img1'].shape
        else:
            pass
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(angle={self.angle})'


@PIPELINES.register_module()
class GaussianNoise:
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

    def __call__(self, results):
        """Call function to add gaussian noise to images. And then clamp the
        images to clamp_range.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        img_keys = get_img_keys(results)

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


@PIPELINES.register_module()
class RandomTranslate:
    """Random translation of the images and flow map.

    .. note: This augmentation is for dense optical flow data, not for sparse
    optical flow data.

    Args:
        prob (float): the probability to do translation.
        x_offset (float | tuple): translate ratio on x axis, randomly choice
            [-x_offset, x_offset] or the given [min, max]. Default: 0.
        y_offset (float | tuple): translate ratio on y axis, randomly choice
            [-x_offset, x_offset] or the given [min, max]. Default: 0.
    """

    def __init__(self, prob=0., x_offset=0., y_offset=0.):
        assert isinstance(prob, float) and prob <= 1. and prob >= 0.
        self.prob = prob
        self.x_offset = self._check_input(x_offset)
        self.y_offset = self._check_input(y_offset)

    def _check_input(self, v):
        value = []
        if isinstance(v, float):
            assert v >= 0. and v <= 1.
            value = [-v, v]
        elif isinstance(v, tuple):
            assert v[0] >= -0.1 and v[1] <= 1.
            value[0] = [v[0], v[1]]
        else:
            raise TypeError('Translate offset should be a single number or a '
                            f'list/tuple with length 2, but got {v}.')
        return np.random.uniform(value[0], value[1])

    def __call__(self, results):
        """Call function to translate the images and optical flow map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Processed results.
        """
        translate = np.random.uniform(0, 1) < self.prob
        if translate:

            h, w, _ = results['img_shape']
            tw = w * self.x_offset
            th = h * self.y_offset
            M = np.float32([[1, 0, tw], [0, 1, th]])
            map_keys = get_map_keys(results)
            for k in map_keys:
                results[k] = cv2.warpAffine(results[k], M, (w, h))

        results['translate'] = translate
        results['translate_offset'] = (self.x_offset, self.y_offset)
        return results


@PIPELINES.register_module()
class RandomGamma:
    """Random gamma correction of images.

    Note: gamma larger than 1 make the shadows darker, while gamma smaller than
    1 make dark regions lighter.

    Args:
        gamma_range(list | tuple): A list or tuple of length 2. Uniformly
            sample gamma from gamma_range. Defaults to (0.7, 1.5).
    """

    def __init__(self, gamma_range=(0.7, 1.5)):

        assert isinstance(gamma_range, (list, tuple))

        assert len(gamma_range) == 2

        assert 0 <= gamma_range[0] <= gamma_range[1]

        self.gamma_range = gamma_range

    def __call__(self, results):
        """Call function to process images using gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """
        img_keys = get_img_keys(results)

        # create new meta 'gamma'
        results['gamma'] = random.uniform(*self.gamma_range)

        for k in img_keys:
            results[k] = adjust_gamma(results[k], results['gamma'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma_range={self.gamma_range})'
