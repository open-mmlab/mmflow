# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmflow.datasets import build_dataloader, build_dataset
from mmflow.models import build_flow_estimator


def parse_args():
    parser = argparse.ArgumentParser(description='MMFlow benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging')
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = [build_dataset(cfg.data.test)]
    data_loader = [
        build_dataloader(
            _dataset,
            **cfg.data.test_dataloader,
            dist=False,
        ) for _dataset in dataset
    ]

    model = build_flow_estimator(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = args.warm_up
    pure_inf_time = 0
    total_iters = args.total_iters

    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader[0]):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(test_mode=True, **data)

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
            break


if __name__ == '__main__':
    main()
