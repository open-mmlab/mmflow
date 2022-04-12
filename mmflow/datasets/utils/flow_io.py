# Copyright (c) OpenMMLab. All rights reserved.

import re
from io import BytesIO
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from numpy import ndarray


def read_flow(name: str) -> np.ndarray:
    """Read flow file with the suffix '.flo'.

    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.

    Args:
        name (str): Optical flow file path.

    Returns:
        ndarray: Optical flow
    """

    with open(name, 'rb') as f:

        header = f.read(4)
        if header.decode('utf-8') != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape(
            (height, width, 2))

    return flow


def write_flow(flow: np.ndarray, flow_file: str) -> None:
    """Write the flow in disk.

    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.

    Args:
        flow (ndarray): The optical flow that will be saved.
        flow_file (str): The file for saving optical flow.
    """

    with open(flow_file, 'wb') as f:
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)


def visualize_flow(flow: np.ndarray, save_file: str = None) -> np.ndarray:
    """Flow visualization function.

    Args:
        flow (ndarray): The flow will be render
        save_dir ([type], optional): save dir. Defaults to None.
    Returns:
        ndarray: flow map image with RGB order.
    """

    # return value from mmcv.flow2rgb is [0, 1.] with type np.float32
    flow_map = np.uint8(mmcv.flow2rgb(flow) * 255.)
    if save_file:
        plt.imsave(save_file, flow_map)
    return flow_map


def render_color_wheel(save_file: str = 'color_wheel.png') -> np.ndarray:
    """Render color wheel.

    Args:
        save_file (str): The saved file name . Defaults to 'color_wheel.png'.

    Returns:
        ndarray: color wheel image.
    """
    x0 = 75
    y0 = 75
    height = 151
    width = 151
    flow = np.zeros((height, width, 2), dtype=np.float32)

    grid_x = np.tile(np.expand_dims(np.arange(width), 0), [height, 1])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])

    grid_x0 = np.tile(np.array([x0]), [height, width])
    grid_y0 = np.tile(np.array([y0]), [height, width])

    flow[:, :, 0] = grid_x - grid_x0
    flow[:, :, 1] = grid_y - grid_y0

    return visualize_flow(flow, save_file)


def read_flow_kitti(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read sparse flow file from KITTI dataset.

    This function is modified from
    https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py.
    Copyright (c) 2020, princeton-vl
    Licensed under the BSD 3-Clause License

    Args:
        name (str): The flow file

    Returns:
        Tuple[ndarray, ndarray]: flow and valid map
    """
    # to specify not to change the image depth (16bit)
    flow = cv2.imread(name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    # flow shape (H, W, 2) valid shape (H, W)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def write_flow_kitti(uv: np.ndarray, filename: str):
    """Write the flow in disk.

    This function is modified from
    https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py.
    Copyright (c) 2020, princeton-vl
    Licensed under the BSD 3-Clause License

    Args:
        uv (ndarray): The optical flow that will be saved.
        filename ([type]): The file for saving optical flow.
    """
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def flow_from_bytes(content: bytes, suffix: str = 'flo') -> ndarray:
    """Read dense optical flow from bytes.

    .. note::
        This load optical flow function works for FlyingChairs, FlyingThings3D,
        Sintel, FlyingChairsOcc datasets, but cannot load the data from
        ChairsSDHom.

    Args:
        content (bytes): Optical flow bytes got from files or other streams.

    Returns:
        ndarray: Loaded optical flow with the shape (H, W, 2).
    """

    assert suffix in ('flo', 'pfm'), 'suffix of flow file must be `flo` '\
        f'or `pfm`, but got {suffix}'

    if suffix == 'flo':
        return flo_from_bytes(content)
    else:
        return pfm_from_bytes(content)


def flo_from_bytes(content: bytes):
    """Decode bytes based on flo file.

    Args:
        content (bytes): Optical flow bytes got from files or other streams.

    Returns:
        ndarray: Loaded optical flow with the shape (H, W, 2).
    """

    # header in first 4 bytes
    header = content[:4]
    if header != b'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    # width in second 4 bytes
    width = np.frombuffer(content[4:], np.int32, 1).squeeze()
    # height in third 4 bytes
    height = np.frombuffer(content[8:], np.int32, 1).squeeze()
    # after first 12 bytes, all bytes are flow
    flow = np.frombuffer(content[12:], np.float32, width * height * 2).reshape(
        (height, width, 2))

    return flow


def pfm_from_bytes(content: bytes) -> np.ndarray:
    """Load the file with the suffix '.pfm'.

    Args:
        content (bytes): Optical flow bytes got from files or other streams.

    Returns:
        ndarray: The loaded data
    """

    file = BytesIO(content)

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.frombuffer(file.read(), endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data[:, :, :-1]


def read_pfm(file: str) -> np.ndarray:
    """Load the file with the suffix '.pfm'.

    This function is modified from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    Copyright (c) 2011, LMB, University of Freiburg.

    Args:
        file (str): The file name will be loaded

    Returns:
        ndarray: The loaded data
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('ascii'))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('ascii').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data[:, :, :-1]
