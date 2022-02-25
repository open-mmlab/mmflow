# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from typing import Sequence

import cv2
import mmcv
from numpy import ndarray

try:
    import imageio
except ImportError:
    imageio = None


def parse_args():
    parser = argparse.ArgumentParser(description='Create GIF for demo')
    parser.add_argument(
        '--frame_dir', type=str, default=None, help='directory of frames')
    parser.add_argument(
        '--resize_factor',
        type=float,
        default=0.5,
        help='resize factor for gif')
    parser.add_argument(
        '--out',
        type=str,
        default='result.gif',
        help='gif path where will be saved')
    parser.add_argument('--duration', default=0.1, help='display interval (s)')

    args = parser.parse_args()
    return args


def create_gif(frames: Sequence[ndarray],
               gif_name: str,
               duration: float = 0.1) -> None:
    """Create gif through imageio.

    Args:
        frames (list[ndarray]): Image frames
        gif_name (str): Saved gif name
        duration (int): Display interval (s). Default: 0.1.
    """
    frames_rgb = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)
    if imageio is None:
        raise RuntimeError('imageio is not installed,'
                           'Please use “pip install imageio” to install')
    imageio.mimsave(gif_name, frames_rgb, 'GIF', duration=duration)


def create_frame(frame_dir: str,
                 resize_factor: float = 0.5) -> Sequence[ndarray]:
    """Create gif frame image through matplotlib.

    Args:
        frame_dir (str): The directory of frames.
        resize_factor (float): The factor for rescaling original images.
            Defaults to 0.5.

    Returns:
        list[ndarray]: List of frames.
    """
    frame_files = list(mmcv.scandir(frame_dir))
    frame_files.sort()
    frame_list = []
    for frame_file in frame_files:
        frame = mmcv.imread(osp.join(frame_dir, frame_file))
        frame_scaled = frame if resize_factor == 1. else mmcv.imrescale(
            frame, scale=resize_factor)
        frame_list.append(frame_scaled)

    return frame_list


def create_video(frames: Sequence[ndarray],
                 out_file: str,
                 duration: float = 0.1) -> None:
    """Create a video to save the optical flow.

    Args:
        frames (list, tuple): Image frames.
        out_file (str): The output file to save visualized flow map.
        duration (int): Display interval (s). Default: 0.1.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1 / duration
    H, W = frames[0].shape[:2]
    video_writer = cv2.VideoWriter(out_file, fourcc, fps, (W, H), True)

    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def main():
    args = parse_args()
    frames = create_frame(args.frame_dir, args.resize_factor)
    if args.out[-3:] == 'gif':
        create_gif(frames, args.out, args.duration)
    else:
        create_video(frames, args.out, args.duration)


if __name__ == '__main__':
    main()
