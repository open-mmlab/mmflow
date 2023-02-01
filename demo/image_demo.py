# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser

from mmengine.utils import mkdir_or_exist

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow
from mmengine.registry import init_default_scope


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img1', help='Image1 file')
    parser.add_argument('img2', help='Image2 file')
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
    # register all modules in mmflow into the registries
    init_default_scope('mmflow')

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img1, args.img2)
    # get prediction from result and convert to np
    pred_flow_fw = result[0].pred_flow_fw.data.permute(1, 2, 0).cpu().numpy()
    # save the results
    mkdir_or_exist(args.out_dir)
    visualize_flow(pred_flow_fw,
                   osp.join(args.out_dir, f'{args.out_prefix}.png'))
    write_flow(pred_flow_fw, osp.join(args.out_dir, f'{args.out_prefix}.flo'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
