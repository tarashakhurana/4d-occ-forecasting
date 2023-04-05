import os
import re
import json
import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

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


def load_sequences(args):
    data_loaders = make_data_loaders(args)

    for horizon in ["3s"]:
        data_loader = data_loaders[horizon]

        for i, batch in enumerate(data_loader):
            # input origin: B x T x 3
            # input points: B x N x 3
            # input tindex: B x N x 1
            # every tindex value stores which timestep the
            # corresponding point in points array belongs to
            input_origin, input_points, input_tindex = batch[1:4]

            # output origin: B x T x 3
            # output points: B x M x 3
            # output tindex: B x M x 1
            # every tindex value stores which timestep the
            # corresponding point in points array belongs to
            output_origin, output_points, output_tindex = batch[4:7]

            # YOUR CODE GOES HERE
            # you may process the data one batch at a time through your model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--argo-root", type=str, required=True)
    data_group.add_argument("--argo-split", type=str, choices=["train", "test", "val"])

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--batch-size", type=int, default=8)
    model_group.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    load_sequences(args)

