# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmflow.datasets.builder import build_dataset
from mmflow.registry import VISUALIZERS
from mmflow.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmseg into the registries
    register_all_modules()

    dataset = build_dataset(cfg.train_dataloader.dataset)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.METAINFO

    progress_bar = ProgressBar(len(dataset))
    for item in dataset:
        data_sample = item['data_sample']
        img1_path = item['data_sample'].metainfo['img1_path']
        img2_path = item['data_sample'].metainfo['img2_path']

        window_name = f'{osp.basename(img1_path)}_{osp.basename(img2_path)}'

        visualizer.add_datasample(
            window_name,
            None,
            data_sample,
            show=not args.not_show,
            wait_time=args.show_interval)

        progress_bar.update()


if __name__ == '__main__':
    main()
