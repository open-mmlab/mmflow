# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import re
from typing import Sequence

from mmengine.utils import mkdir_or_exist
from utils import get_data_filename

# these files contain nan, so exclude them.
exclude_files = dict(
    left_into_future=[
        '0004573.flo',
        '0006336.flo',
        '0016948.flo',
        '0015148.flo',
        '0006922.flo',
        '0003147.flo',
        '0003149.flo',
        '0000879.flo',
        '0006337.flo',
        '0014658.flo',
        '0015748.flo',
        '0001717.flo',
        '0000119.flo',
        '0017578.flo',
        '0004118.flo',
        '0004117.flo',
        '0004304.flo',
        '0004154.flo',
        '0011530.flo',
    ],
    right_into_future=[
        '0006336.flo',
        '0003148.flo',
        '0004117.flo',
        '0003666.flo',
    ],
    left_into_past=[
        '0000162.flo',
        '0004705.flo',
        '0006878.flo',
        '0004876.flo',
        '0004045.flo',
        '0000053.flo',
        '0005055.flo',
        '0000163.flo',
        '0000161.flo',
        '0000121.flo',
        '0000931.flo',
        '0005054.flo',
    ],
    right_into_past=[
        '0006878.flo',
        '0003147.flo',
        '0001549.flo',
        '0000053.flo',
        '0005034.flo',
        '0003148.flo',
        '0005055.flo',
        '0000161.flo',
        '0001648.flo',
        '0000160.flo',
        '0005054.flo',
    ])


def parse_args():
    parser = argparse.ArgumentParser(
        description='FlyingThings3D_subset dataset preparation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/FlyingThings3D_subset',
        help='Directory for dataset.')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/FlyingThings3D_subset/',
        help='Directory to save the annotation files for '
        'FlyingThings3D subset dataset')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    flow_fw_dir = 'into_future'
    flow_bw_dir = 'into_past'
    flow_suffix = '.flo'
    occ_suffix = '.png'
    img_suffix = '.png'

    def _revise_dir(flow_fw_filename) -> Sequence[str]:
        """Revise directory of optical flow to get the directories of image and
        occlusion mask.

        Args:
            flow_fw_filename (str):  abstract file path of optical flow from
                image1 to image2.
        Returns:
            Sequence[str]: abstract paths of image1, image2, forward
                occlusion mask, backward occlusion mask and backward optical
                flow.
        """

        idx_f = int(osp.splitext(osp.basename(flow_fw_filename))[0])

        img1_filename = flow_fw_filename.replace(
            f'{os.sep}flow{os.sep}', f'{os.sep}image_clean{os.sep}')
        img1_filename = img1_filename.replace(f'{os.sep}into_future{os.sep}',
                                              f'{os.sep}')

        img1_filename = img1_filename.replace(flow_suffix, img_suffix)
        img2_filename = re.sub(r'\d{7}', f'{idx_f+1:07d}', img1_filename)

        flow_bw_filename = flow_fw_filename.replace(
            f'{os.sep}into_future{os.sep}', f'{os.sep}into_past{os.sep}')
        flow_bw_filename = re.sub(r'\d{7}', f'{idx_f+1:07d}', flow_bw_filename)

        occ_fw_filename = flow_fw_filename.replace(
            f'{os.sep}flow{os.sep}', f'{os.sep}flow_occlusions{os.sep}')

        occ_fw_filename = occ_fw_filename.replace(flow_suffix, occ_suffix)
        occ_bw_filename = flow_bw_filename.replace(
            f'{os.sep}flow{os.sep}', f'{os.sep}flow_occlusions{os.sep}')

        occ_bw_filename = occ_bw_filename.replace(flow_suffix, occ_suffix)

        return (img1_filename, img2_filename, occ_fw_filename, occ_bw_filename,
                flow_bw_filename)

    train_list = []
    test_list = []
    for subset in ('train', 'val'):
        data_root = osp.join(args.data_root, subset)

        flow_root = osp.join(data_root, 'flow')

        all_scene = ['left', 'right']
        for scene in all_scene:
            flow_dir = osp.join(flow_root, scene)

            flow_fw_dir = osp.join(flow_dir, 'into_future')
            flow_bw_dir = osp.join(flow_dir, 'into_past')

            tmp_flow_fw_filenames = get_data_filename(
                flow_fw_dir, None, exclude_files[scene + '_into_future'])
            tmp_flow_bw_filenames = get_data_filename(
                flow_bw_dir, None, exclude_files[scene + '_into_past'])

            for ii in range(len(tmp_flow_fw_filenames)):
                flo_fw = tmp_flow_fw_filenames[ii]
                img1, img2, occ_fw, occ_bw, flo_bw = _revise_dir(flo_fw)
                if (not (flo_bw in tmp_flow_bw_filenames)
                        or not osp.isfile(img1) or not osp.isfile(img2)
                        or not osp.isfile(occ_fw) or not osp.isfile(occ_bw)):
                    continue
                data_info = dict(
                    scene=scene,
                    img1_path=img1,
                    img2_path=img2,
                    flow_fw_path=flo_fw,
                    flow_bw_path=flo_bw,
                    occ_fw_path=occ_fw,
                    occ_bw_path=occ_bw)
                if subset == 'train':
                    train_list.append(data_info)
                else:
                    test_list.append(data_info)
    mkdir_or_exist(args.save_dir)
    with open(osp.join(args.save_dir, 'train.json'), 'w') as jsonfile:
        json.dump({'data_list': train_list, 'metainfo': {}}, jsonfile)
    with open(osp.join(args.save_dir, 'test.json'), 'w') as jsonfile:
        json.dump({'data_list': test_list, 'metainfo': {}}, jsonfile)


if __name__ == '__main__':
    main()
