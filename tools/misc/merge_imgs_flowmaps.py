# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import cv2
import mmcv
import numpy as np

try:
    import imageio
except ImportError:
    imageio = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge images and visualized flow')
    parser.add_argument(
        '--img_dir', type=str, default=None, help='directory of images')
    parser.add_argument(
        '--flow_dir',
        type=str,
        default=None,
        help='directory of visualized flow')
    parser.add_argument(
        '--resize_factor',
        type=float,
        default=0.5,
        help='resize factor for gif')
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='directory to save merged results')

    args = parser.parse_args()
    return args


def merge_imgs_flow(img_dir: str, flow_dir: str, out_dir: str) -> None:
    """Load images and visualized flow maps and merge them.

    Args:
        img_dir ([str): The directory of images.
        flow_dir (str): The directory of flow maps.
        out_dir (str): The directory to save the frames
    """
    img_files = list(mmcv.scandir(img_dir))
    flow_files = list(mmcv.scandir(flow_dir))
    img_files.sort()
    flow_files.sort()
    # img is longer than flow
    for i in range(len(img_files) - 1):
        img = mmcv.imread(osp.join(img_dir, img_files[i]))
        flow = mmcv.imread(osp.join(flow_dir, flow_files[i]))
        frame = np.concatenate((img, flow), axis=1)

        cv2.imwrite(osp.join(out_dir, flow_files[i]), frame)


def main():
    args = parse_args()
    merge_imgs_flow(args.img_dir, args.flow_dir, args.out_dir)


if __name__ == '__main__':
    main()
