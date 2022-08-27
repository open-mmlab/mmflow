# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp

from mmengine.utils import mkdir_or_exist
from utils import get_data_filename


def parse_args():
    parser = argparse.ArgumentParser(description='Sintel dataset preparation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/Sintel',
        help='Directory for dataset.')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/Sintel/',
        help='Directory to save the annotation files for Sintel dataset')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    all_pass = ['clean', 'final']

    def _get_filenames(data_dir, data_suffix, img_idx=None):
        data_filenames = []

        data_filenames = get_data_filename(data_dir, data_suffix)
        data_filenames.sort()
        if img_idx == 1:
            return data_filenames[:-1]
        elif img_idx == 2:
            return data_filenames[1:]
        else:
            return data_filenames

    def _get_data_list(subset_dir):
        """Get the paths for images and optical flow."""
        img_suffix = '.png'
        flow_suffix = '.flo'
        occ_suffix = '.png'
        invalid_suffix = '.png'

        data_root = osp.join(args.data_root, subset_dir)
        if subset_dir == 'testing':
            data_root = osp.join(data_root, 'test')
        flow_root = osp.join(data_root, 'flow')
        occ_root = osp.join(data_root, 'occlusions')
        invalid_root = osp.join(data_root, 'invalid')

        data_list = []

        for p in all_pass:
            img_root = osp.join(data_root, p)
            scene = os.listdir(img_root)
            for s in scene:
                img1_dir = osp.join(img_root, s)
                img2_dir = osp.join(img_root, s)

                flow_dir = osp.join(flow_root, s)
                occ_dir = osp.join(occ_root, s)
                invalid_dir = osp.join(invalid_root, s)
                img1_filenames = _get_filenames(img1_dir, img_suffix, 1)
                img2_filenames = _get_filenames(img2_dir, img_suffix, 2)
                if subset_dir == 'training':
                    flow_filenames = _get_filenames(flow_dir, flow_suffix)
                    occ_filenames = _get_filenames(occ_dir, occ_suffix)
                    invalid_filenames = _get_filenames(invalid_dir,
                                                       invalid_suffix)
                else:
                    flow_filenames = [None] * len(img1_filenames)
                    occ_filenames = [None] * len(img1_filenames)
                    invalid_filenames = [None] * len(img1_filenames)

                for i_img1, i_img2, i_flow, i_occ, i_invalid in zip(
                        img1_filenames, img2_filenames, flow_filenames,
                        occ_filenames, invalid_filenames):
                    data_info = dict(
                        pass_style=p,
                        scene=s,
                        img1_path=i_img1,
                        img2_path=i_img2,
                        flow_fw_path=i_flow,
                        invalid_path=i_invalid,
                        occ_fw_path=i_occ)
                    data_list.append(data_info)
        mkdir_or_exist(args.save_dir)
        if subset_dir == 'training':
            annotation_file = osp.join(args.save_dir, 'train.json')
        else:
            annotation_file = osp.join(args.save_dir, 'test.json')
        with open(annotation_file, 'w') as jsonfile:
            json.dump({'data_list': data_list, 'metainfo': {}}, jsonfile)

    _get_data_list('training')
    _get_data_list('testing')


if __name__ == '__main__':
    main()
