# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import time

import numpy as np
import torch
from mmengine import Config
from mmengine.fileio import dump
from mmengine.runner import Runner, load_checkpoint
from mmengine.utils import mkdir_or_exist

from mmflow.models import build_flow_estimator
from mmflow.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='MMFlow benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the results will be dumped '
              'into the directory as json'))
    parser.add_argument(
        '--warm_up',
        type=int,
        default=5,
        help='the warm-up iterations should be skipped')
    parser.add_argument(
        '--total_iters',
        type=int,
        default=200,
        help='the total iterations during benchmark')
    parser.add_argument('--repeat-times', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.work_dir is not None:
        mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, f'fps_{timestamp}.json')
    else:
        # use config filename as default work_dir if cfg.work_dir is None
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mkdir_or_exist(osp.abspath(work_dir))
        json_file = osp.join(work_dir, f'fps_{timestamp}.json')

    repeat_times = args.repeat_times
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False

    benchmark_dict = dict(config=args.config, unit='img / s')
    overall_fps_list = []
    if isinstance(cfg.test_dataloader, list):
        test_dataloader = cfg.test_dataloader[0]
    else:
        test_dataloader = cfg.test_dataloader
    test_dataloader.batch_size = 1
    for time_index in range(repeat_times):
        print(f'Run {time_index + 1}:')
        # build the dataloader
        data_loader = Runner.build_dataloader(test_dataloader)

        model = build_flow_estimator(cfg.model)

        if 'checkpoint' in args and osp.exists(args.checkpoint):
            load_checkpoint(model, args.checkpoint, map_location='cpu')

        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        # the first several iterations may be very slow so skip them
        num_warmup = args.warm_up
        pure_inf_time = 0
        total_iters = args.total_iters

        # benchmark with 200 image and take the average
        for i, data in enumerate(data_loader):
            data = model.data_preprocessor(data, True)

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                model(**data, mode='predict')

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed
                if (i + 1) % args.log_interval == 0:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Done image [{i + 1:<3}/ {total_iters}], '
                          f'fps: {fps:.2f} img / s')

            if (i + 1) == total_iters:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Overall fps: {fps:.2f} img / s')
                benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                overall_fps_list.append(fps)
                break

    benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
    benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
    print(f'Average fps of {repeat_times} evaluations: '
          f'{benchmark_dict["average_fps"]}')
    print(f'The variance of {repeat_times} evaluations: '
          f'{benchmark_dict["fps_variance"]}')
    dump(benchmark_dict, json_file, indent=4)


if __name__ == '__main__':
    main()
