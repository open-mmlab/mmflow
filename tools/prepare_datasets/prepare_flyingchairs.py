# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='FlyingChairs dataset preparation')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/FlyingChairs_release',
        help='Directory for dataset.')
    parser.add_argument(
        '--split-file',
        type=str,
        default='data/FlyingChairs_release/FlyingChairs_train_val.txt',
        help='File name of '
        'train-validation split file for FlyingChairs.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    split = np.loadtxt(args.split_file, dtype=np.int32).tolist()
    # unpack FlyingChairs directly, will see `data` subdirctory.
    img1_dir = osp.join(args.data_root, 'data')
    img2_dir = osp.join(args.data_root, 'data')
    flow_dir = osp.join(args.data_root, 'data')

    # data in FlyingChairs dataset has specific suffix
    img1_suffix = '_img1.ppm'
    img2_suffix = '_img2.ppm'
    flow_suffix = '_flow.flo'

    img1_filenames = [f for f in mmcv.scandir(img1_dir, suffix=img1_suffix)]
    img2_filenames = [f for f in mmcv.scandir(img2_dir, suffix=img2_suffix)]
    flow_filenames = [f for f in mmcv.scandir(flow_dir, suffix=flow_suffix)]
    img1_filenames.sort()
    img2_filenames.sort()
    flow_filenames.sort()

    train_list = []
    test_list = []
    train_meta = dict(dataset='FlyingChairs', subset='train')
    test_meta = dict(dataset='FlyingChairs', subset='test')

    for i, flag in enumerate(split):

        data_info = dict(
            img1_dir='data',
            img2_dir='data',
            flow_dir='data',
            img_info=dict(
                filename1=img1_filenames[i], filename2=img2_filenames[i]),
            ann_info=dict(filename_flow=flow_filenames[i]))

        if flag == 1:
            train_list.append(data_info)
        else:
            test_list.append(data_info)

    with open('FlyingChairs_train.json', 'w') as jsonfile:
        json.dump({'data_list': train_list, 'metainfo': train_meta}, jsonfile)

    with open('FlyingChairs_test.json', 'w') as jsonfile:
        json.dump({'data_list': test_list, 'metainfo': test_meta}, jsonfile)


if __name__ == '__main__':
    main()
