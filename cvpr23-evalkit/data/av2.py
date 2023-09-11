import os
import yaml
import numpy as np
import time
import torch
import time
import pathlib
from torch.utils.data import Dataset

import av2.utils.io as io_utils
from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3

from data.common import get_argoverse2_split


def invert_transform(rigid_trans):
    R = rigid_trans[:3, :3]
    t = rigid_trans[:3,  3]
    R_inv = R.T
    inv_rigid_trans = np.eye(4, dtype=float)
    inv_rigid_trans[:3, :3] = R_inv
    inv_rigid_trans[:3,  3] = - R_inv @ t
    return inv_rigid_trans


class Argoverse2Dataset(Dataset):
    SPLIT_SEQUENCES = get_argoverse2_split()
    SPLIT_SEQUENCES['trainval'] = SPLIT_SEQUENCES['train'] + SPLIT_SEQUENCES['val']
    SPLIT_SEQUENCES['all'] = SPLIT_SEQUENCES['train'] + SPLIT_SEQUENCES['val'] + SPLIT_SEQUENCES['test']

    # TODO: add classes later
    def __init__(self, argo_root, argo_split, kwargs):

        # setup pairs of input frames and output frames
        self.argo_root = argo_root
        self.argo_split = argo_split

        self.n_input = kwargs["n_input"]
        self.input_step = kwargs["input_step"]
        self.n_output = kwargs["n_output"]
        self.output_step = kwargs["output_step"]

        assert(self.input_step == self.output_step)

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.poses = []
        self.valid_indices = []
        self.lidar_to_egovehicle = []

        start_time = time.time()

        for log_id in self.SPLIT_SEQUENCES[argo_split]:
            # calibration file per sequence
            calibpath = os.path.join(argo_root, argo_split, log_id, "calibration/egovehicle_SE3_sensor.feather")
            calib_df = io_utils.read_feather(calibpath)
            calib = calib_df.loc[calib_df['sensor_name'] == "up_lidar"]
            lidar_to_egovehicle = convert_pose_dataframe_to_SE3(calib)

            posepath = os.path.join(argo_root, argo_split, log_id, 'city_SE3_egovehicle.feather')
            poses_df = io_utils.read_feather(posepath)

            velo_dir = os.path.join(argo_root, argo_split, log_id, "sensors/lidar/")
            if not os.path.exists(velo_dir):
                raise RuntimeError("Velodyne directory missing: " + velo_dir)

            velo_names = sorted(os.listdir(velo_dir))
            start_index = len(self.filenames)
            # if len(velo_names) < (self.n_input * self.input_step + self.n_output * self.output_step * 2):
            #     continue
            for idx in range(0, len(velo_names), self.input_step):
                velo_name = velo_names[idx]
                self.sequences.append(log_id)
                self.filenames.append(velo_name)
                timestamp = int(velo_name[:-8])
                pose = poses_df.loc[poses_df['timestamp_ns'] == timestamp]
                egovehicle_to_city = convert_pose_dataframe_to_SE3(pose)
                # assuming that as soon as the lidar point cloud will be loaded, it will be transformed
                # to the lidar coordinate frame, inside __get_item__
                relative_pose = egovehicle_to_city.transform_matrix @ lidar_to_egovehicle.transform_matrix
                self.poses.append(relative_pose)
                self.lidar_to_egovehicle.append(lidar_to_egovehicle.transform_matrix)

            #
            end_index = len(self.filenames)
            #
            valid_start_index = start_index + self.n_input  # (self.n_input // 10)
            valid_end_index = end_index - self.n_output
            self.valid_indices += list(range(valid_start_index, valid_end_index))

        assert(len(self.sequences) == len(self.filenames))
        print("Total number of sequences found", len(self.valid_indices))
        print("Time taken to load all the Argoverse 2.0 data", time.time() - start_time)

    def __len__(self):
        return len(self.valid_indices)

    def filter_out_ego(self, pts):
        # 2020 Ford Fusion Hybrid Dimensions
        # 192″ L x 75″ W x 58″ H
        # 4.87 m x 1.90 m x 1.47 m
        xx, yy, zz = pts[:, :3].T
        ego_mask = np.logical_and(
            np.logical_and(-1.25 <= yy, yy <= 1.25),
            np.logical_and(-1.75 <= xx, xx <= 3.75)
        )
        return pts[np.logical_not(ego_mask), :]

    def __getitem__(self, idx):
        # do most of the heavy lifting (alignment across frames etc.)
        ref_index = self.valid_indices[idx]
        ref_sequence = self.sequences[ref_index]
        ref_filename = self.filenames[ref_index]

        # reference frame's global pose
        ref_pose = self.poses[ref_index]
        inv_ref_pose = invert_transform(ref_pose)
        #
        first_index = ref_index - (self.n_input - 1)
        last_index = ref_index + self.n_output

        # total length: self.n_input + (self.n_output - 1)
        # ref_index appears once in indices
        indices = [*range(first_index, ref_index + 1)] + \
            [*range(ref_index + 1, last_index + 1)]

        #
        input_points_list, input_tindex_list, input_origin_list = [], [], []
        output_points_list, output_tindex_list, output_origin_list = [], [], []


        for i, index in enumerate(indices):
            # valid frame
            if 0 <= index and index < len(self.sequences) and self.sequences[index] == ref_sequence:
                sequence = self.sequences[index]
                velo_name = self.filenames[index]

                scan_path = os.path.join(self.argo_root, self.argo_split, sequence, "sensors/lidar/", velo_name)
                scan = io_utils.read_lidar_sweep(scan_path)

                points = np.ones((scan.shape[0], 4))
                points[:, :3] = scan[:, :3]
                lidar_to_egovehicle = self.lidar_to_egovehicle[index]
                points = (invert_transform(lidar_to_egovehicle) @ points.T).T

                # remove returns from the ego vehicle
                points = self.filter_out_ego(points)

                #
                pose = self.poses[index]
                tf = inv_ref_pose @ pose

                #
                origin_tf = tf[:3, 3].astype(np.float32)
                points_tf = (tf @ points.T).T
                points_tf = points_tf[:, :3].astype(np.float32)

            else:
                origin_tf = np.array([0, 0, 0], dtype=np.float32)
                points_tf = np.full((0, 3), float('nan'), dtype=np.float32)

            if i < self.n_input:
                tindex = np.full(len(points_tf), self.n_input - i - 1, dtype=np.float32)
                input_origin_list.append(origin_tf)
                input_points_list.append(points_tf)
                input_tindex_list.append(tindex)
            else:
                tindex = np.full(len(points_tf), i - self.n_input, dtype=np.float32)
                output_origin_list.append(origin_tf)
                output_points_list.append(points_tf)
                output_tindex_list.append(tindex)

        input_origin_tensor = torch.from_numpy(np.stack(input_origin_list))
        input_points_tensor = torch.from_numpy(np.concatenate(input_points_list))
        input_tindex_tensor = torch.from_numpy(np.concatenate(input_tindex_list))

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))
        output_points_tensor = torch.from_numpy(np.concatenate(output_points_list))
        output_tindex_tensor = torch.from_numpy(np.concatenate(output_tindex_list))

        return (ref_sequence, ref_filename), \
            input_origin_tensor, input_points_tensor, input_tindex_tensor, \
            output_origin_tensor, output_points_tensor, output_tindex_tensor

