import torch
import numpy as np

# A CUSTOMIZED COLLATION FUNCTION TO ACCOMODATE THE FACT THAT
# WE HAVE A DIFFERENT NUMBER OF POINTS FROM EVERY FRAME
# WE USE PADDING TO MAKE SURE THEY CAN BE BATCHED AS A 2D MATRIX
def CollateFn(batch):
    #
    filenames = [item[0] for item in batch]

    # input origin: N x T x 3
    input_origin = torch.stack([item[1] for item in batch])

    # input points: N x max_{i} (sum_{t} n_{t, i}) x 3
    max_n_input_points = max([len(item[2]) for item in batch])
    input_points = torch.stack(
        [
            torch.nn.functional.pad(
                item[2],
                (0, 0, 0, max_n_input_points - len(item[2])),
                mode="constant",
                value=float("nan"),
            )
            for item in batch
        ]
    )

    # input tindex: N x max_{i} (sum_{t} n_{t, i})
    input_tindex = torch.stack(
        [
            torch.nn.functional.pad(
                item[3],
                (0, max_n_input_points - len(item[3])),
                mode="constant",
                value=-1,
            )
            for item in batch
        ]
    )

    # output origin: N x T x 3
    output_origin = torch.stack([item[4] for item in batch])

    # output points: N x max_{i} (sum_{t} n_{t, i}) x 3
    max_n_output_points = max([len(item[5]) for item in batch])

    output_points = torch.stack(
        [
            torch.nn.functional.pad(
                item[5],
                (0, 0, 0, max_n_output_points - len(item[5])),
                mode="constant",
                value=float("nan"),
            )
            for item in batch
        ]
    )

    # output tindex: N x max_{i} (sum_{t} n_{t, i})
    output_tindex = torch.stack(
        [
            torch.nn.functional.pad(
                item[6],
                (0, max_n_output_points - len(item[6])),
                mode="constant",
                value=-1,
            )
            for item in batch
        ]
    )

    return (
        filenames,
        input_origin,
        input_points,
        input_tindex,
        output_origin,
        output_points,
        output_tindex,
    )

def get_argoverse2_split():
    from av2.datasets.sensor.splits import TRAIN, TEST, VAL
    return {'train': TRAIN, 'val': VAL, 'test': TEST}

