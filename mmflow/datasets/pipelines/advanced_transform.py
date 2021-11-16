# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional, Sequence

import cv2
import numpy as np

from ..builder import PIPELINES
from .transforms import get_flow_keys, get_img_keys, get_valid_keys


def get_occ_keys(results: dict) -> Sequence[str]:
    """Get occlusion key in result.

    Args:
        results (dict): data with dict type in data augmentation pipeline.
    Returns:
        list: [description]
    """
    occ_keys = []
    if 'ann_fields' in results:
        ann_keys = copy.deepcopy(results['ann_fields'])
        for k in ann_keys:
            if k.find('occ') > -1:
                occ_keys.append(k)
    return occ_keys


def theta_is_valid(theta: np.ndarray) -> bool:
    """Whether affine transform theta is a valid affine transform.

    A valid affine transform is an affine transform which guarantees the
    transformed image covers the whole original picture frame.

    Args:
        theta (ndarray): affine transform matrix.
    Returns:
        bool: whether this transform matrix is valid.
    """

    bounds = np.array([
        [-0.5, -0.5, 1.],  # left top
        [-0.5, 0.5, 1.],  # left bottom
        [0.5, -0.5, 1.],  # right top
        [0.5, 0.5, 1.],  # right bottom
    ])
    """
    (-0.5, -0.5)          (0.5, -0.5)
                 --------
                |        |
                |        |
                |        |
                 --------
    (-0.5, 0.5)          (0.5, 0.5)
    """
    bounds = (np.linalg.inv(theta) @ bounds.T).T

    valid = ((bounds[:, :2] >= -0.5) & (bounds[:, :2] <= 0.5)).all()

    return valid


def check_out_of_bound(flow: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """Check pixels that will move out of bound after warping by flow and mark
    as occluded pixels.

    Revise occlusion mask for transformed optical flow data.

    Args:
        flow (ndarray): optical flow data.
        occ (ndarray): original occlusion mask.
    Returns:
        ndarray: the occlusion mask for optical flow.
    """

    height, width, _ = flow.shape

    xx, yy = np.meshgrid(range(width), range(height))

    xx = xx.astype(flow.dtype)
    yy = yy.astype(flow.dtype)

    xx += flow[:, :, 0]
    yy += flow[:, :, 1]

    out_of_bound = ((xx < 0) | (yy < 0) | (xx >= width) |
                    (yy >= height)).astype(occ.dtype)

    occ = np.clip(out_of_bound + occ, 0, 1)

    return occ


def transform_img(img: np.ndarray, theta: np.ndarray, height: int,
                  width: int) -> np.ndarray:
    """Transform image with cv2 warpAffine.

    Args:
        img (ndarray): image that will be transformed.
        theta (ndarray): transform matrix.
        height (int): height of output image.
        width (int): width of output image.

    Returns:
        ndarray: transformed image.
    """
    return cv2.warpAffine(img, theta[:2, :], (width, height))


def transform_flow(flow: np.ndarray, valid: np.ndarray, theta1: np.ndarray,
                   theta2: np.ndarray, height: int, width: int) -> np.ndarray:
    """Transform optical flow with cv2 warpAffine.

    Args:
        flow (ndarray): flow that will be transformed.
        theta1 (ndarray): global transform matrix.
        theta2 (ndarray): relative transform matrix.
        height (int): height of output image.
        width (int): width of output image.

    Returns:
        ndarray: transformed optical flow.
    """

    flow_ = cv2.warpAffine(flow, theta1[:2, :], (width, height))
    if valid is not None:
        flow_ = flow_ / (valid[:, :, None] + 1e-12)
    """
    X1                 Affine(theta1)             X1'
               x                                   x
    theta1(-1) y           ->                      y
               1                                   1

    X2                 Affine(theta2)             X2'
               x   u                                         x   u
    theta1(-1) y + v       ->           theta2 x {theta1(-1) y + v}
               1   0                                         1   0
                                        flow' = X2' -X1'
    """

    # (u, v) -> (u, v, 0); shape (height, width, 2) -> (height, width, 3)
    homo_flow_ = np.concatenate((flow_, np.zeros((height, width, 1))), axis=2)

    xx, yy = np.meshgrid(range(width), range(height))

    # grid of homogeneous coordinates
    homo_grid = np.stack((xx, yy, np.ones((height, width))),
                         axis=2).astype(flow.dtype)
    """
    theta2 x [u, v, 0]T + (theta2 x theta1(-1) - [1, 1, 1]) x [x, y, 1]T
    """
    flow_final = homo_grid @ (theta2 @ np.linalg.inv(theta1) -
                              np.eye(3)).T + homo_flow_ @ theta2.T

    return flow_final[:, :, :2]


@PIPELINES.register_module()
class RandomAffine:
    """Random affine transformation of images, flow map and occlusion map (if
    available).

    Keys of global_transform and relative_transform should be the subset of
    ('translates', 'zoom', 'shear', 'rotate'). And also, each key and its
    corresponding values has to satisfy the following rules:
        - translates: the translation ratios along x axis and y axis. Defaults
            to(0., 0.).
        - zoom: the min and max zoom ratios. Defaults to (1.0, 1.0).
        - shear: the min and max shear ratios. Defaults to (1.0, 1.0).
        - rotate: the min and max rotate degree. Defaults to (0., 0.).

    Args:
        global_transform (dict): A dict which contains keys: transform, zoom,
            shear, rotate. global_transform will transform both img1 and img2.
        relative_transform (dict): A dict which contains keys: transform, zoom,
            shear, rotate. relative_transform will only transform img2 after
            global_transform to both images.
        preserve_valid (bool): Whether continue transforming until both images
            are valid. A valid affine transform is an affine transform which
            guarantees the transformed image covers the whole original picture
            frame. Defaults to True.
        check_bound (bool): Whether to check out of bound for transformed
            occlusion maps. If True, all pixels in borders of img1 but not in
            borders of img2 will be marked occluded. Defaults to False.
    """

    def __init__(self,
                 global_transform: Optional[dict] = None,
                 relative_transform: Optional[dict] = None,
                 preserve_valid: bool = True,
                 check_bound: bool = False) -> None:

        self.DEFAULT_TRANSFORM = dict(
            translates=(0., 0.),
            zoom=(1.0, 1.0),
            shear=(1.0, 1.0),
            rotate=(0., 0.))

        self.global_transform = self._check_input(global_transform)
        self.relative_transform = self._check_input(relative_transform)

        assert isinstance(preserve_valid, bool)
        self.preserve_valid = preserve_valid

        assert isinstance(check_bound, bool)
        self.check_bound = check_bound

    def _check_input(self, transform: dict) -> dict:
        """Check whethere input transform.

        Args:
            transform (dict): A dict which may contains keys: transform, zoom,
                shear, rotate. If transform miss some key, it will be set the
                default value.

        Returns:
            dict: transform dict with all valid values.
        """

        ret = dict() if not isinstance(transform, dict) else transform.copy()

        assert set(ret).issubset(self.DEFAULT_TRANSFORM), (
            f'Got unexpected keys in {transform}. \n'
            f"Valid keys should be the subset of ('translates', 'zoom', "
            f"'shear', 'rotate')")

        for k in self.DEFAULT_TRANSFORM:
            if k not in ret:
                ret[k] = self.DEFAULT_TRANSFORM[k]

            assert isinstance(ret[k], (list, tuple))
            assert len(ret[k]) == 2
            assert ret[k][0] <= ret[k][1]

        return ret

    def __call__(self, results: dict) -> dict:
        """

        Args:
            results (dict): data including image, annotation and meta
                information in data augmentation pipeline.

        Returns:
            dict: transformed data.
        """

        h, w, _ = results['img_shape']

        # theta0_ndc, theta1_ndc and theta2_ndc are 3 x 3 affine transformation
        # matrix in normal device coordinates, with origin at the center of
        # pictures and picture's width range and height range from [-0.5, 0.5]
        # and [-0.5, 0.5].
        theta0_ndc = np.identity(3)

        # apply global transform to identity matrix theta0_ndc
        theta1_ndc = self._apply_random_affine_to_theta(
            theta0_ndc, **self.global_transform)

        # apply relative transform to theta1_ndc
        theta2_ndc = self._apply_random_affine_to_theta(
            theta1_ndc, **self.relative_transform)

        # T is similar transform matrix
        T = np.array([[1. / (w - 1.), 0., -0.5], [0., 1. / (h - 1.), -0.5],
                      [0., 0., 1.]], np.float32)

        T_inv = np.linalg.inv(T)

        # theta1_world and theta2_world are affine transformations in world
        # coordinates, with origin at top left corner of pictures and picture's
        # width range and height range from [0, width] and [0, height].
        theta1_world = T_inv @ theta1_ndc @ T
        theta2_world = T_inv @ theta2_ndc @ T
        theta_world_li = [theta1_world, theta2_world]

        img_keys = get_img_keys(results)
        flow_keys = get_flow_keys(results)
        occ_keys = get_occ_keys(results)
        valid_keys = get_valid_keys(results)

        # transform img1 and img2
        for i in range(len(img_keys)):
            results[img_keys[i]] = transform_img(results[img_keys[i]],
                                                 theta_world_li[i], h, w)

        # transform flows
        for i in range(len(flow_keys)):
            if len(valid_keys) == len(flow_keys):
                valid = results[valid_keys[i]]

                results[valid_keys[i]] = transform_img(results[valid_keys[i]],
                                                       theta_world_li[i], h, w)
                results[flow_keys[i]] = transform_flow(
                    flow=results[flow_keys[i]] * valid[:, :, None],
                    valid=results[valid_keys[i]],
                    theta1=theta_world_li[i],
                    theta2=theta_world_li[1 - i],
                    height=h,
                    width=w)

            else:
                results[flow_keys[i]] = transform_flow(
                    flow=results[flow_keys[i]],
                    valid=None,
                    theta1=theta_world_li[i],
                    theta2=theta_world_li[1 - i],
                    height=h,
                    width=w)

        # transform occlusion if available
        for i in range(len(occ_keys)):
            results[occ_keys[i]] = transform_img(results[occ_keys[i]],
                                                 theta_world_li[i], h, w)
            if self.check_bound:
                results[occ_keys[i]] = check_out_of_bound(
                    results[flow_keys[i]], results[occ_keys[i]])

        # create new meta 'global_ndc_affine_mat'
        results['global_ndc_affine_mat'] = theta1_ndc

        # create new meta 'relative_ndc_affine_mat'
        results['relative_ndc_affine_mat'] = theta2_ndc

        return results

    def _apply_random_affine_to_theta(self, theta: np.ndarray,
                                      translates: Sequence[float],
                                      zoom: Sequence[float],
                                      shear: Sequence[float],
                                      rotate: Sequence[float]) -> np.ndarray:
        """Get the 3 x 3 affine transformation matrix in normal device
        coordinates based on input transformation matrix and transformation
        dict.

        Args:
            translates (list): the translation ratios along x axis and y axis.
                Defaults to(0., 0.).
            zoom (list): the min and max zoom ratios. Defaults to (1.0, 1.0).
            shear (list): the min and max shear ratios. Defaults to (1.0, 1.0).
            rotate (list): the min and max rotate degree. Defaults to (0., 0.).

        Returns:
            ndarray: affine transformation matrix.
        """

        valid = False

        while not valid:

            zoom_ = np.random.uniform(zoom[0], zoom[1])
            shear_ = np.random.uniform(shear[0], shear[1])

            t_x = np.random.uniform(-translates[0], translates[0])
            t_y = np.random.uniform(-translates[1], translates[1])

            phi = np.random.uniform(rotate[0] * np.pi / 180.,
                                    rotate[1] * np.pi / 180.)

            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            translate_mat = np.array([
                [1., 0., t_x],
                [0., 1., t_y],
                [0., 0., 1.],
            ])

            rotate_mat = np.array([
                [cos_phi, -sin_phi, 0.],
                [sin_phi, cos_phi, 0.],
                [0., 0., 1.],
            ])

            shear_mat = np.array([
                [shear_, 0., 0.],
                [0., 1. / shear_, 0.],
                [0., 0., 1.],
            ])

            zoom_mat = np.array([
                [zoom_, 0., 0.],
                [0., zoom_, 0.],
                [0., 0., 1.],
            ])

            T = translate_mat @ rotate_mat @ shear_mat @ zoom_mat

            theta_propose = T @ theta

            if not self.preserve_valid:
                break

            valid = theta_is_valid(theta_propose)

        return theta_propose

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(global_transform={self.global_transform}, '
                f'relative_transform={self.relative_transform}, '
                f'preserve_valid={self.preserve_valid}, '
                f'check_bound={self.check_bound})')
