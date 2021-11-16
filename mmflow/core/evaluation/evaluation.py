# Copyright (c) OpenMMLab. All rights reserved.
import sys
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Union

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info

from .metrics import eval_metrics

Module = torch.nn.Module
DataLoader = torch.utils.data.DataLoader


def online_evaluation(model: Module,
                      data_loader: DataLoader,
                      metric: Union[str, Sequence[str]] = 'EPE',
                      **kwargs: Any) -> Dict[str, np.ndarray]:
    """Evaluate model online.

    Args:
        model (nn.Module): The optical flow estimator model.
        data_loader (DataLoader): The test dataloader.
        metric (str, list): Metrics to be evaluated. Default: 'EPE'.
        kwargs (any): Evaluation arguments fed into the evaluate function of
            the dataset.
    Returns:
        dict: The evaluation result.
    """

    if isinstance(model, MMDistributedDataParallel):
        return multi_gpu_online_evaluation(model, data_loader, metric=metric)
    else:
        return single_gpu_online_evaluation(
            model, data_loader, metric=metric, **kwargs)


def single_gpu_online_evaluation(
        model: Module,
        data_loader: DataLoader,
        metric: Union[str, Sequence[str]] = 'EPE') -> Dict[str, np.ndarray]:
    """Evaluate model with single gpu online.

    This function will not save the flow. Namely, there do not exist any IO
    operations in this function. Thus, in general, `online` mode will achieve a
    faster evaluation. However, using this function, the `img_metas` must
    include the ground truth e.g. `flow_gt` or `flow_fw_gt` and `flow_bw_gt`.

    Args:
        model (nn.Module): The optical flow estimator model.
        data_loader (DataLoader): The test dataloader.
        metric(str, list): Metrics to be evaluated. Default: 'EPE'.

    Returns:
        dict: The evaluation result.
    """

    model.eval()
    metrics = metric if isinstance(metric, (type, list)) else [metric]
    result_metrics = defaultdict(list)

    prog_bar = mmcv.ProgressBar(len(data_loader))
    for data in data_loader:
        with torch.no_grad():
            batch_results = model(test_mode=True, **data)
            img_metas = data['img_metas'].data[0]
            batch_flow = []
            batch_flow_gt = []
            batch_valid = []
            # a batch of result and a batch of img_metas
            for i in range(len(batch_results)):
                result = batch_results[i]
                img_meta = img_metas[i]

                # result.keys() is 'flow' or ['flow_fw','flow_bw']
                # img_meta.keys() is 'flow_gt' or ['flow_fw_gt','flow_bw_gt']
                for k in result.keys():

                    if img_meta.get(k + '_gt', None) is None:
                        # img_meta does not have flow_bw_gt, so just check
                        # the forward predicted.
                        if k == 'flow_bw':
                            continue
                        elif k == 'flow_fw':
                            batch_flow_gt.append(img_meta['flow_gt'])
                    else:
                        batch_flow_gt.append(img_meta[k + '_gt'])

                    batch_flow.append(result[k])
                    batch_valid.append(
                        img_meta.get('valid', np.ones_like(result[k][..., 0])))

            batch_results_metrics = eval_metrics(batch_flow, batch_flow_gt,
                                                 batch_valid, metrics)
            for i_metric in metrics:
                result_metrics[i_metric].append(
                    batch_results_metrics[i_metric])

            prog_bar.update()

    for i_metric in metrics:
        if result_metrics.get(i_metric) is None:
            raise KeyError(f'Model cannot compute {i_metric}')
        result_metrics[i_metric] = np.array(result_metrics[i_metric]).mean()

    return result_metrics


def multi_gpu_online_evaluation(
        model: Module,
        data_loader: DataLoader,
        metric: Union[str, Sequence[str]] = 'EPE',
        tmpdir: Optional[str] = None,
        gpu_collect: bool = False) -> Dict[str, np.ndarray]:
    """Evaluate model with multiple gpus online.

    This function will not save the flow. Namely, there do not exist any IO
    operations in this function. Thus, in general, `online` mode will achieve a
    faster evaluation. However, using this function, the `img_metas` must
    include the ground truth e.g. `flow_gt` or `flow_fw_gt` and `flow_bw_gt`.

    Args:
        model (nn.Module): The optical flow estimator model.
        data_loader (DataLoader): The test dataloader.
        metric(str, list): Metrics to be evaluated. Default: 'EPE'.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        dict: The evaluation result.
    """

    model.eval()
    metrics = metric if isinstance(metric, (type, list)) else [metric]
    result_metrics = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for data in data_loader:
        with torch.no_grad():
            batch_results = model(test_mode=True, **data)
            # data['img_metas'] is Datacontainer
            img_metas = data['img_metas'].data[0]
            batch_flow = []
            batch_flow_gt = []
            batch_valid = []

            # a batch of result and a batch of img_metas
            for i in range(len(batch_results)):
                result = batch_results[i]
                img_meta = img_metas[i]
                # result.keys() is 'flow' or ['flow_fw','flow_bw']
                # img_meta.keys() is 'flow_gt' or ['flow_fw_gt','flow_bw_gt']
                for k in result.keys():

                    if img_meta.get(k + '_gt', None) is None:
                        # img_meta does not have flow_bw_gt, so just check
                        # the forward predicted.
                        if k == 'flow_bw':
                            continue
                        elif k == 'flow_fw':
                            batch_flow_gt.append(img_meta['flow_gt'])
                    else:
                        batch_flow_gt.append(img_meta[k + '_gt'])

                    batch_flow.append(result[k])
                    batch_valid.append(
                        img_meta.get('valid', np.ones_like(result[k][..., 0])))

            batch_results_metrics = eval_metrics(batch_flow, batch_flow_gt,
                                                 batch_valid, metrics)
            # result_metrics is different from result_metrics in
            # `single_gpu_online_evaluation`
            # result_metrics is Sequence[Dict[str,ndarray]]
            result_metrics.append(batch_results_metrics)

        if rank == 0:
            batch_size = len(batch_results)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    from mmflow.apis.test import collect_results_cpu, collect_results_gpu
    if gpu_collect:
        result_metrics = collect_results_gpu(result_metrics, len(dataset))
    else:
        result_metrics = collect_results_cpu(result_metrics, len(dataset),
                                             tmpdir)
    rank, world_size = get_dist_info()
    if rank == 0:
        sys.stdout.write('\n')
        # result_metrics_ is final result of evaluation with type
        # dict(metric_name=metric)
        result_metrics_ = dict()

        for sample_result_metrics in result_metrics:
            for k in sample_result_metrics.keys():
                if result_metrics_.get(k, None) is None:
                    result_metrics_[k] = sample_result_metrics[k] / len(
                        result_metrics)
                else:
                    result_metrics_[k] += sample_result_metrics[k] / len(
                        result_metrics)

        return result_metrics_
