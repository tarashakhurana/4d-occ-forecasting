import time
import json
import torch
import random
import argparse
import numpy as np
import chamferdist
from tqdm import tqdm
from chamferdist import ChamferDistance
from pathlib import Path
from typing import Any, Dict
from urllib.request import urlretrieve

PC_RANGE = [-70.0, -70.0, -4.5, 70.0, 70.0, 4.5]
chamfer_distance = ChamferDistance()

def compute_nearfield_chamferdistance(gt_pcd, pred_pcd):
    """
    Arguments:
        gt_pcd  : np array of shape N x 3
        pred_pcd: np array of shape N x 3
    """
    mask1 = torch.logical_and(PC_RANGE[0] <= pred_pcd[:, 0], pred_pcd[:, 0] <= PC_RANGE[3])
    mask2 = torch.logical_and(PC_RANGE[1] <= pred_pcd[:, 1], pred_pcd[:, 1] <= PC_RANGE[4])
    mask3 = torch.logical_and(PC_RANGE[2] <= pred_pcd[:, 2], pred_pcd[:, 2] <= PC_RANGE[5])
    inner_mask_pred = mask1 & mask2 & mask3

    mask1 = torch.logical_and(PC_RANGE[0] <= gt_pcd[:, 0], gt_pcd[:, 0] <= PC_RANGE[3])
    mask2 = torch.logical_and(PC_RANGE[1] <= gt_pcd[:, 1], gt_pcd[:, 1] <= PC_RANGE[4])
    mask3 = torch.logical_and(PC_RANGE[2] <= gt_pcd[:, 2], gt_pcd[:, 2] <= PC_RANGE[5])
    inner_mask_gt = mask1 & mask2 & mask3

    if inner_mask_pred.sum() == 0 or inner_mask_gt.sum() == 0:
        return 0.0

    pred_pcd_inner = pred_pcd[inner_mask_pred]
    gt_pcd_inner = gt_pcd[inner_mask_gt]

    cd_forward, cd_backward, CD_info = chamfer_distance(
        pred_pcd_inner[None, ...].float(),
        gt_pcd_inner[None, ...].float(),
        bidirectional=True,
        reduction='sum')

    chamfer_dist_value = (cd_forward / pred_pcd_inner.shape[0]) + (cd_backward / gt_pcd_inner.shape[0])
    return chamfer_dist_value / 2.0


def compute_chamferdistance(gt_pcd, pred_pcd):
    """
    Arguments:
        gt_pcd  : np array of shape N x 3
        pred_pcd: np array of shape N x 3
    """

    # print("Reached right before chamfer dist call", torch.from_numpy(pred_pcd)[None, ...].shape)

    cd_forward, cd_backward, CD_info = chamfer_distance(
        pred_pcd[None, ...].float(),
        gt_pcd[None, ...].float(),
        bidirectional=True,
        reduction='sum')

    # print("everything", everything)

    chamfer_dist_value = (cd_forward / pred_pcd.shape[0]) + (cd_backward / gt_pcd.shape[0])
    return chamfer_dist_value / 2.0



def evaluate(
        annotation_file: str, submission_file: str) -> Dict[str, Any]:
    print("Starting Evaluation.....")

    annotations = json.load(open(annotation_file, "rb"))
    predictions = json.load(open(submission_file, "rb"))

    assert annotations["queries"][0]["horizon"] == predictions["queries"][0]["horizon"]

    gt_rays = annotations["queries"][0]["rays"]
    pd_rays = predictions["queries"][0]["rays"]

    l1_error = 0.0
    absrel_error = 0.0
    cd = 0.0
    near_field_cd = 0.0
    count = 0

    for log in tqdm(gt_rays):
        # make sure predictions don't skip any log
        assert log in pd_rays
        gt_log = gt_rays[log]
        pd_log = pd_rays[log]

        for sequence in gt_log:
            # make sure predictions don't skip any sequence
            assert sequence in pd_log

            # print("Doing", log, sequence)

            gt_seq = gt_log[sequence]
            pd_seq = pd_log[sequence]

            for t in range(len(gt_seq)):

                gt_seq_t = torch.from_numpy(np.array(gt_seq[t])).to("cuda:0")
                pd_seq_t = torch.from_numpy(np.array(pd_seq[t])).to("cuda:0")

                gt_depth = gt_seq_t[:, -1]
                pd_depth = pd_seq_t[:, -1]
                ray_orgs = gt_seq_t[:, :3]
                ray_dirs = gt_seq_t[:, 3:6]

                assert gt_depth.shape[0] == pd_depth.shape[0]

                # get the first metric: l1 error
                l1_error += torch.abs(gt_depth - pd_depth).mean()

                # get the second metric: absrel error
                absrel_error += (torch.abs(gt_depth - pd_depth) / gt_depth).mean()

                # make point clouds for third and fourth metrics
                gt_pcd = ray_orgs + ray_dirs * gt_depth[..., None]
                pd_pcd = ray_orgs + ray_dirs * pd_depth[..., None]

                # get the third metric: chamfer distance
                cd += compute_chamferdistance(gt_pcd, pd_pcd)

                # get the fourth metric: near-field chamfer distance
                near_field_cd += compute_nearfield_chamferdistance(gt_pcd, pd_pcd)

                # update the count
                count += 1

    l1_error /= count
    absrel_error /= count
    cd /= count
    near_field_cd /= count


    output = {
            "L1": float(l1_error.cpu().numpy()),
            "AbsRel": float(absrel_error.cpu().numpy()),
            "NFCD": float(near_field_cd.cpu().numpy()),
            "CD": float(cd.cpu().numpy()),
    }
    print("Completed evaluation for CVPR23 Phase")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation", type=str,
                            required=True)
    parser.add_argument("--submission", type=str,
                            required=True)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    output = evaluate(args.annotation, args.submission)

    print("Results:", output)

