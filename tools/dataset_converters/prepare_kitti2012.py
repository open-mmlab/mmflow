# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

from mmengine.utils import mkdir_or_exist
from utils import get_data_filename


def parse_args():
    parser = argparse.ArgumentParser(
        description='KITTI2012 dataset preparation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/kitti2012',
        help='Directory for dataset.')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/kitti2012/',
        help='Directory to save the annotation files for KITTI2012 dataset')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    subset_dir = 'training'

    args.data_root = osp.join(args.data_root, subset_dir)
    # In KITTI 2012, data in `colored_0` is original data
    img1_dir = osp.join(args.data_root, 'colored_0')
    img2_dir = osp.join(args.data_root, 'colored_0')
    flow_dir = osp.join(args.data_root, 'flow_occ')

    img1_suffix = '_10.png',
    img2_suffix = '_11.png',
    flow_suffix = '_10.png'

    img1_filenames = get_data_filename(img1_dir, img1_suffix)
    img2_filenames = get_data_filename(img2_dir, img2_suffix)
    flow_filenames = get_data_filename(flow_dir, flow_suffix)

    data_list = []
    for i_img1, i_img2, i_flow_fw in zip(img1_filenames, img2_filenames,
                                         flow_filenames):
        data_info = dict(
            img1_path=i_img1, img2_path=i_img2, flow_fw_path=i_flow_fw)
        data_list.append(data_info)

    mkdir_or_exist(args.save_dir)
    annotation_file = osp.join(args.save_dir, 'train.json')
    with open(annotation_file, 'w') as jsonfile:
        json.dump({'data_list': data_list, 'metainfo': {}}, jsonfile)


if __name__ == '__main__':
    main()
