# Based on Monodepth2

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_depth_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths

    
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_ate(gtruth_xyz, pred_xyz, return_pred_xyz_scaled=False):
    
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz[0]
    pred_xyz += offset[None,:]

    # Optimize the scaling factor
    scale = torch.sum(gtruth_xyz * pred_xyz)/torch.sum(pred_xyz ** 2)
    pred_xyz_scaled = pred_xyz * scale
    alignment_error = pred_xyz_scaled - gtruth_xyz
    rmse = torch.sqrt(torch.sum(alignment_error ** 2))/gtruth_xyz.shape[0]
    if return_pred_xyz_scaled:
        return rmse, pred_xyz_scaled
    else:
        return rmse
