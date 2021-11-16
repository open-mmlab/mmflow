# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Sequence

import cv2
import numpy as np
from numpy import ndarray

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow

try:
    import imageio
except ImportError:
    imageio = None


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video', help='video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('out', help='File to save visualized flow map')
    parser.add_argument(
        '--gt',
        default=None,
        help='video file of ground truth for input video')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):

    assert args.out[-3:] == 'gif' or args.out[-3:] == 'mp4', \
        f'Output file must be gif and mp4, but got {args.out[-3:]}.'

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # load video
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f'Failed to load video file {args.video}'
    # get video info
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    imgs = []
    while (cap.isOpened()):
        # Get frames
        flag, img = cap.read()
        if not flag:
            break
        imgs.append(img)

    gts = []
    if args.gt is not None:

        cap_gt = cv2.VideoCapture(args.gt)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while (cap_gt.isOpened()):
            flag, gt = cap_gt.read()
            if not flag:
                break
            gt = cv2.putText(gt, 'Ground Truth', (20, 50), font, 1,
                             (255, 255, 255), 3, cv2.LINE_AA)
            gts.append(gt)

        assert len(gts) == len(
            imgs) - 1, 'Ground truth length doesn\'t match video frames'

    frame_list = []

    for i in range(len(imgs) - 1):
        img1 = imgs[i]
        img2 = imgs[i + 1]
        # estimate flow
        result = inference_model(model, img1, img2)
        flow_map = visualize_flow(result, None)
        # visualize_flow return flow map with RGB order
        flow_map = cv2.cvtColor(flow_map, cv2.COLOR_RGB2BGR)
        if len(gts) > 0:
            frame = np.concatenate((flow_map, gts[i]), axis=1)
        else:
            frame = flow_map
        frame_list.append(frame)

    size = (frame_list[0].shape[1], frame_list[0].shape[0])
    cap.release()

    if args.out[-3:] == 'gif':
        create_gif(frame_list, args.out, 1 / fps)
    else:
        create_video(frame_list, args.out, fourcc, fps, size)


def create_video(frames: Sequence[ndarray], out: str, fourcc: int, fps: int,
                 size: tuple) -> None:
    """Create a video to save the optical flow.

    Args:
        frames (list, tuple): Image frames.
        out (str): The output file to save visualized flow map.
        fourcc (int): Code of codec used to compress the frames.
        fps (int):      Framerate of the created video stream.
        size (tuple): Size of the video frames.
    """
    # init video writer
    video_writer = cv2.VideoWriter(out, fourcc, fps, size, True)

    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def create_gif(frames: Sequence[ndarray],
               gif_name: str,
               duration: float = 0.1) -> None:
    """Create gif through imageio.

    Args:
        frames (list[ndarray]): Image frames.
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


if __name__ == '__main__':
    args = parse_args()
    main(args)
