import os
import re
import json
import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict

from data.common import CollateFn
from data.av2 import Argoverse2Dataset


def make_data_loaders(args):
    dataset_1s_kwargs = {
        "n_input": 5,
        "input_step": 2,
        "n_output": 5,
        "output_step": 2,
    }
    dataset_3s_kwargs = {
        "n_input": 5,
        "input_step": 6,
        "n_output": 5,
        "output_step": 6,
    }
    data_loader_kwargs = {
        "pin_memory": False,
        "shuffle": False,
        "drop_last": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }

    data_loaders = {
            "1s": DataLoader(
                Argoverse2Dataset(args.argo_root, args.argo_split, dataset_1s_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            ),
            "3s": DataLoader(
                Argoverse2Dataset(args.argo_root, args.argo_split, dataset_3s_kwargs),
                collate_fn=CollateFn,
                **data_loader_kwargs,
            )
    }

    return data_loaders


def generate_and_save(args):
    data_loaders = make_data_loaders(args)
    query_rays_json = {"queries": []}

    for horizon in ["3s"]:
        data_loader = data_loaders[horizon]

        rays = defaultdict(lambda: defaultdict(list))
        for i, batch in enumerate(data_loader):
            # output origin: B x T x 3
            # output points: B x M x 3
            # output tindex: B x M x 1
            # every tindex value stores which timestep the
            # corresponding point in points array belongs to
            output_origin, output_points, output_tindex = batch[4:7]
            filenames = batch[0]

            for j, entry in enumerate(batch):
                origin = output_origin[j]
                points = output_points[j]
                tindex = output_tindex[j]
                fname  = filenames[j]
                num_timesteps = origin.shape[0]

                per_timestep_points = []
                for t in range(num_timesteps):
                    points_t  = points[tindex == t]
                    origin_t  = np.repeat(origin[t][None, :], points_t.shape[0], axis=0)
                    direction = points_t - origin_t
                    unit_direction = direction / np.sqrt((direction ** 2).sum(axis=1, keepdims=True))
                    origin_direction = np.hstack((origin_t, unit_direction)).astype(float)
                    num_points = origin_direction.shape[0]
                    origin_direction_subsampled = origin_direction[::5, ...]
                    per_timestep_points.append(origin_direction_subsampled.tolist())

                rays[fname[0]][fname[1]] = per_timestep_points

        queries = defaultdict()
        queries["horizon"] = horizon
        queries["rays"] = rays

        query_rays_json["queries"].append(queries)

        with open("query_rays.json", "w") as f:
            json.dump(query_rays_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--argo-root", type=str,
                            required=True)
    data_group.add_argument("--argo-split", type=str,
                            choices=["train", "test", "val"],
                            required=True)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--batch-size", type=int, default=8)
    model_group.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    generate_and_save(args)

