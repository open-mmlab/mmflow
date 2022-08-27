# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

from mmengine.utils import mkdir_or_exist
from utils import get_data_filename


def parse_args():
    parser = argparse.ArgumentParser(
        description='ChairsSDHom dataset preparation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/ChairsSDHom',
        help='Directory for dataset.')

    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/ChairsSDHom/',
        help='Directory to save the annotation files for ChairsSDHom dataset')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # these files contain nan, so exclude them.
    exclude_files = ['08755.pfm']

    img1_suffix = '.png'
    img2_suffix = '.png'
    flow_suffix = '.pfm'

    def _get_data_list(subset_dir):
        data_root = osp.join(args.data_root, 'data', subset_dir)
        img1_dir = osp.join(data_root, 't0')
        img2_dir = osp.join(data_root, 't1')
        flow_dir = osp.join(data_root, 'flow')

        flow_filenames = get_data_filename(
            flow_dir, flow_suffix, exclude=exclude_files)
        img1_filenames, img2_filenames = _revise_dir(flow_filenames, img1_dir,
                                                     img2_dir)

        data_list = []

        for i in range(len(flow_filenames)):
            data_info = dict(
                img1_path=img1_filenames[i],
                img2_path=img2_filenames[i],
                flow_fw_path=flow_filenames[i])
            data_list.append(data_info)
        mkdir_or_exist(args.save_dir)
        with open(osp.join(args.save_dir, f'{subset_dir}.json'),
                  'w') as jsonfile:
            json.dump({
                'metainfo': {},
                'data_list': data_list
            },
                      jsonfile,
                      indent=4)

    def _revise_dir(flow_filenames, img1_dir, img2_dir):
        img1_filenames = []
        img2_filenames = []
        for flow_filename in flow_filenames:
            idx = int(osp.splitext(osp.basename(flow_filename))[0])
            img1_filename = osp.join(img1_dir, f'{idx:05d}' + img1_suffix)
            img2_filename = osp.join(img2_dir, f'{idx:05d}' + img2_suffix)
            img1_filenames.append(img1_filename)
            img2_filenames.append(img2_filename)
        return img1_filenames, img2_filenames

    _get_data_list('train')
    _get_data_list('test')


if __name__ == '__main__':
    main()
