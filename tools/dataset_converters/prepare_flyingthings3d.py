# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp
from glob import glob

from mmengine.utils import mkdir_or_exist
from utils import get_data_filename


def parse_args():
    parser = argparse.ArgumentParser(
        description='FlyingThings3D dataset preparation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/flyingthings3d',
        help='Directory for dataset.')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/flyingthings3d/',
        help='Directory to save the annotation files for '
        'FlyingThings3D dataset')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    flow_fw_dir = 'into_future'
    flow_bw_dir = 'into_past'
    flow_suffix = '.pfm'
    img_suffix = '.png'
    scene = ['left', 'right']

    def _get_data_list(subset_dir):
        imgs_dirs = []
        flow_fw_dirs = []
        flow_bw_dirs = []
        for i_scene in scene:
            tmp_imgs_dirs = glob(
                osp.join(args.data_root, 'frames_cleanpass',
                         subset_dir + '/*/*'))
            tmp_imgs_dirs = [osp.join(f, i_scene) for f in tmp_imgs_dirs]
            imgs_dirs += tmp_imgs_dirs
            tmp_flow_fw_dirs = glob(
                osp.join(args.data_root, 'optical_flow',
                         subset_dir + '/*/*/' + flow_fw_dir))

            tmp_flow_fw_dirs = [osp.join(f, i_scene) for f in tmp_flow_fw_dirs]
            flow_fw_dirs += tmp_flow_fw_dirs

            tmp_flow_bw_dirs = glob(
                osp.join(args.data_root, 'optical_flow',
                         subset_dir + '/*/*/' + flow_bw_dir))
            tmp_flow_bw_dirs = [osp.join(f, i_scene) for f in tmp_flow_bw_dirs]
            flow_bw_dirs += tmp_flow_bw_dirs
        img1_filenames = []
        img2_filenames = []
        flow_fw_filenames = []
        flow_bw_filenames = []
        for idir, fw_dir, bw_dir in zip(imgs_dirs, flow_fw_dirs, flow_bw_dirs):

            img_filenames_ = get_data_filename(idir, img_suffix)
            img_filenames_.sort()
            flow_fw_filenames_ = get_data_filename(fw_dir, flow_suffix)
            flow_fw_filenames_.sort()
            flow_bw_filenames_ = get_data_filename(bw_dir, flow_suffix)
            flow_bw_filenames_.sort()

            img1_filenames.extend(img_filenames_[:-1])
            img2_filenames.extend(img_filenames_[1:])
            flow_fw_filenames.extend(flow_fw_filenames_[:-1])
            flow_bw_filenames.extend(flow_bw_filenames_[1:])

        data_list = []

        assert len(img1_filenames) == len(img2_filenames) == len(
            flow_fw_filenames) == len(flow_bw_filenames)

        for i_img1, i_img2, i_flow_fw, i_flow_bw in zip(
                img1_filenames, img2_filenames, flow_fw_filenames,
                flow_bw_filenames):

            i_scene = scene[0] if scene[0] in i_img1 else scene[1]

            data_info = dict(
                scene=i_scene,
                pass_style='clean',
                img1_path=i_img1,
                img2_path=i_img2,
                flow_fw_path=i_flow_fw,
                flow_bw_path=i_flow_bw,
            )
            data_info_final = dict(
                scene=i_scene,
                pass_style='final',
                img1_path=i_img1.replace('clean', 'final'),
                img2_path=i_img2.replace('clean', 'final'),
                flow_fw_path=i_flow_fw,
                flow_bw_path=i_flow_bw,
            )

            data_list.append(data_info)
            data_list.append(data_info_final)

        mkdir_or_exist(args.save_dir)
        if subset_dir == 'TRAIN':
            annotation_file = osp.join(args.save_dir, 'train.json')
        else:
            annotation_file = osp.join(args.save_dir, 'test.json')
        with open(annotation_file, 'w') as jsonfile:
            json.dump({'data_list': data_list, 'metainfo': {}}, jsonfile)

    _get_data_list('TRAIN')
    _get_data_list('TEST')


if __name__ == '__main__':
    main()
