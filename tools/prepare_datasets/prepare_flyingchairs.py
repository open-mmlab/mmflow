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
        if flag == 1:
            train_list.append(
                dict(
                    img1_filename=img1_filenames[i],
                    img2_filename=img2_filenames[i],
                    flow_filename=flow_filenames[i]))

        else:
            test_list.append(
                dict(
                    img1_filename=img1_filenames[i],
                    img2_filename=img2_filenames[i],
                    flow_filename=flow_filenames[i]))
    train_json = json.dumps({'data_list': train_list, 'metainfo': train_meta})
    test_json = json.dumps({'data_list': test_list, 'metainfo': test_meta})
    print(test_json)
    with open('FlyingChairs_train.json', 'w') as jsonfile:
        json.dump(train_json, jsonfile)

    with open('FlyingChairs_test.json', 'w') as jsonfile:
        json.dump(test_json, jsonfile)


if __name__ == '__main__':
    main()
