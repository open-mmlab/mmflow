# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp
from glob import glob

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(description='HD1K dataset preparation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/hd1k',
        help='Directory for dataset.')

    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/hd1k/',
        help='Directory to save the annotation files for HD1K dataset')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    img1_dir = osp.join(args.data_root, 'hd1k_input/image_2')
    flow_dir = osp.join(args.data_root, 'hd1k_flow_gt/flow_occ')

    img1_filenames = []
    img2_filenames = []
    flow_filenames = []

    seq_ix = 0
    while True:
        flows = sorted(glob(osp.join(flow_dir, '%06d_*.png' % seq_ix)))
        images = sorted(glob(osp.join(img1_dir, '%06d_*.png' % seq_ix)))

        if len(flows) == 0:
            break

        for i in range(len(flows) - 1):
            flow_filenames += [flows[i]]
            img1_filenames += [images[i]]
            img2_filenames += [images[i + 1]]

        seq_ix += 1

    train_list = []

    for i in range(len(flow_filenames)):
        data_info = dict(
            img1_path=osp.join(img1_filenames[i]),
            img2_path=osp.join(img2_filenames[i]),
            flow_fw_path=osp.join(flow_filenames[i]))

        train_list.append(data_info)
    mkdir_or_exist(args.save_dir)
    with open(osp.join(args.save_dir, 'train.json'), 'w') as jsonfile:
        json.dump({'metainfo': {}, 'data_list': train_list}, jsonfile)


if __name__ == '__main__':
    main()
