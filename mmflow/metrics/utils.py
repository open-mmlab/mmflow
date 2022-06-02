# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np


def end_point_error_map(flow_pred: np.ndarray,
                        flow_gt: np.ndarray) -> np.ndarray:
    """Calculate end point error map.

    Args:
        flow_pred (ndarray): The predicted optical flow with the
            shape (H, W, 2).
        flow_gt (ndarray): The ground truth of optical flow with the shape
            (H, W, 2).

    Returns:
        ndarray: End point error map with the shape (H , W).
    """
    return np.sqrt(np.sum((flow_pred - flow_gt)**2, axis=-1))
