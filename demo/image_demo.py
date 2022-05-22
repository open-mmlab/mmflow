# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser

import mmcv

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img1', help='Image1 file')
    parser.add_argument('img2', help='Image2 file')
    parser.add_argument(
        '--valid',
        help='Valid file. If the predicted flow is'
        'sparse, valid mask will filter the output flow map.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        'out_dir', help='Path of directory to save flow map and flow file')
    parser.add_argument(
        '--out_prefix',
        help='The prefix for the output results '
        'including flow file and visualized flow map',
        default='flow')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img1, args.img2, valids=args.valid)
    # save the results
    mmcv.mkdir_or_exist(args.out_dir)
    visualize_flow(result, osp.join(args.out_dir, f'{args.out_prefix}.png'))
    write_flow(result, osp.join(args.out_dir, f'{args.out_prefix}.flo'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
