import os
import yaml

import numpy as np

import torch
from torch.utils.data import Dataset

from data.common import KittiPoint2nuScenes, KittiPoints2nuScenes

# https://github.com/PRBonn/semantic-kitti-api/blob/master/generate_sequential.py#L14


def parse_calibration(filename):
    """ read calibration file with given filename
      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename
      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(Tr_inv @ (pose @ Tr))

    return poses


class KittiDataset(Dataset):
    SPLIT_SEQUENCES = {
        "train": ["00", "01", "02", "03", "04", "05"],
        "val": ["06", "07"],
        "test": ["08", "09", "10"],
        "plot": ["08"],
        "trainval": ["00", "01", "02", "03", "04", "05", "06", "07"],
        "all": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    }
    NUM_CLASS = 20

    def __init__(self, kitti_root, kitti_cfg, kitti_split, kwargs):

        # setup pairs of input frames and output frames
        self.kitti_root = kitti_root
        self.info = yaml.safe_load(open(kitti_cfg))

        self.pc_range = kwargs["pc_range"]
        self.voxel_size = kwargs["voxel_size"]

        self.n_input = kwargs["n_input"]
        self.input_step = kwargs["input_step"]
        self.n_output = kwargs["n_output"]
        self.output_step = kwargs["output_step"]

        # NOTE:
        self.sequences = []
        self.filenames = []
        self.poses = []

        for sequence in self.SPLIT_SEQUENCES[kitti_split]:
            # calibration file per sequence
            calib_path = os.path.join(kitti_root, "sequences", sequence, "calib.txt")
            calib = parse_calibration(calib_path)

            # one pose file, many lines, one line per frame
            pose_path = os.path.join(kitti_root, "sequences", sequence, "poses.txt")
            poses = parse_poses(pose_path, calib)
            self.poses += poses

            velo_dir = os.path.join(kitti_root, "sequences", sequence, "velodyne")
            if not os.path.exists(velo_dir):
                raise RuntimeError("Velodyne directory missing: " + velo_dir)

            velo_names = sorted(os.listdir(velo_dir))
            for velo_name in velo_names:
                self.sequences.append(sequence)
                self.filenames.append(velo_name)

        assert(len(self.sequences) == len(self.filenames))

    def __len__(self):
        return len(self.filenames)

    def filter_by_range(self, pts):
        # nuscenes x, y, z
        xx, yy, zz = pts[:, :3].T
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        x_mask = np.logical_and(x_min <= xx, xx < x_max)
        y_mask = np.logical_and(y_min <= yy, yy < y_max)
        z_mask = np.logical_and(z_min <= zz, zz < z_max)
        mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
        return pts[mask, :]

    def filter_out_ego(self, pts):
        # kitti x, y, z
        # 2010 Volkswagen Passat Wagon/Dimensions
        # 188″ L x 72″ W x 60″ H
        # 4.78 m x 1.83 m x 1.53 m
        xx, yy, zz = pts[:, :3].T
        ego_mask = np.logical_and(
            np.logical_and(-1.55 <= yy, yy <= 1.55),
            np.logical_and(-2.0 <= xx, xx <= 3.5)
        )
        return pts[np.logical_not(ego_mask), :]

    def __getitem__(self, idx):
        # do most of the heavy lifting (alignment across frames etc.)
        ref_index = idx

        #
        ref_sequence = self.sequences[ref_index]
        ref_filename = self.filenames[ref_index]

        # reference frame's global pose
        ref_pose = self.poses[ref_index]
        inv_ref_pose = np.linalg.inv(ref_pose)

        #
        first_index = ref_index - (self.n_input - 1) * self.input_step
        last_index = ref_index + self.n_output * self.output_step

        indices = [*range(first_index, ref_index + 1, self.input_step)] + \
            [*range(ref_index + self.output_step, last_index + 1, self.output_step)]

        #
        input_points_list, input_tindex_list, input_origin_list = [], [], []
        output_points_list, output_tindex_list, output_origin_list = [], [], []
        for i, index in enumerate(indices):
            # valid frame
            if 0 <= index and index < len(self.sequences) and self.sequences[index] == ref_sequence:
                sequence = self.sequences[index]
                velo_name = self.filenames[index]

                scan_path = os.path.join(self.kitti_root, "sequences", sequence, "velodyne", velo_name)
                scan = np.fromfile(scan_path, dtype=np.float32)
                scan = scan.reshape((-1, 4))

                points = np.ones((scan.shape))
                points[:, :3] = scan[:, :3]

                # remove returns from the ego vehicle
                points = self.filter_out_ego(points)

                #
                pose = self.poses[index]
                tf = inv_ref_pose @ pose
                origin_tf = tf[:3, 3].astype(np.float32)
                origin_tf = KittiPoint2nuScenes(origin_tf)

                #
                points_tf = (tf @ points.T).T
                points_tf = KittiPoints2nuScenes(points_tf)
                points_tf = points_tf.astype(np.float32)

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

        displacement = torch.from_numpy(input_origin_list[0] - input_origin_list[1])

        # FIXED
        return (ref_sequence, ref_filename, displacement), \
            input_points_tensor, input_tindex_tensor, \
            output_origin_tensor, output_points_tensor, output_tindex_tensor

