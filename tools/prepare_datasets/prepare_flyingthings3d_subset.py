# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import re
from typing import Optional, Sequence, Union

import mmcv

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

    args = parser.parse_args()

    return args


def get_data_filename(
        data_dirs: Union[Sequence[str], str],
        suffix: Optional[str] = None,
        exclude: Optional[Sequence[str]] = None) -> Sequence[str]:
    """Get file name from data directory.

    Args:
        data_dirs (list, str): the directory of data
        suffix (str, optional): the suffix for data file. Defaults to None.
        exclude (list, optional): list of files will be excluded.

    Returns:
        list: the list of data file.
    """

    if data_dirs is None:
        return None
    data_dirs = data_dirs \
        if isinstance(data_dirs, (list, tuple)) else [data_dirs]

    suffix = '' if suffix is None else suffix
    print(exclude)
    if exclude is None:
        exclude = []
    else:
        assert isinstance(exclude, (list, tuple))

    files = []
    for data_dir in data_dirs:
        for f in mmcv.scandir(data_dir, suffix=suffix):
            if f not in exclude:
                files.append(osp.join(data_dir, f))
    files.sort()
    return files


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
    train_meta = dict(dataset='FlyingThings3D_subset', subset='train')
    test_meta = dict(dataset='FlyingThings3D_subset', subset='test')
    for subset in ('train', 'val'):
        data_root = osp.join(args.data_root, subset)

        img_root = osp.join(data_root, 'image_clean')
        flow_root = osp.join(data_root, 'flow')
        occ_root = osp.join(data_root, 'flow_occlusions')

        all_scene = ['left', 'right']
        for scene in all_scene:
            img1_dir = osp.join(img_root, scene)
            img2_dir = osp.join(img_root, scene)
            flow_dir = osp.join(flow_root, scene)
            occ_dir = osp.join(occ_root, scene)

            flow_fw_dir = osp.join(flow_dir, 'into_future')
            flow_bw_dir = osp.join(flow_dir, 'into_past')
            occ_fw_dir = osp.join(occ_dir, 'info_future')
            occ_bw_dir = osp.join(occ_dir, 'info_past')

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
                    img1_dir=img1_dir,
                    img2_dir=img2_dir,
                    flow_fw_dir=flow_fw_dir,
                    flow_bw_dir=flow_bw_dir,
                    occ_fw_dir=occ_fw_dir,
                    occ_bw_dir=occ_bw_dir,
                    img_info=dict(filename1=img1, filename2=img2),
                    ann_info=dict(
                        filename_flow_fw=flo_fw,
                        filename_flow_bw=flo_bw,
                        filename_occ_fw=occ_fw,
                        filename_occ_bw=occ_bw))
                if subset == 'train':
                    train_list.append(data_info)
                else:
                    test_list.append(data_info)
    with open('FlyingThings3D_subset_train.json', 'w') as jsonfile:
        json.dump({'data_list': train_list, 'metainfo': train_meta}, jsonfile)
    with open('FlyingThings3D_subset_test.json', 'w') as jsonfile:
        json.dump({'data_list': test_list, 'metainfo': test_meta}, jsonfile)


if __name__ == '__main__':
    main()
