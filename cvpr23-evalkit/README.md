# Argoverse 2 4D Occupancy Forecasting Evaluation Kit

Official helper scripts for the 2023 Argoverse 2.0 4D Occupancy Forecasting Challenge
at the CVPR 2023 Workshop on Autonomous Driving.

## Installation

This evaluation kit only requires the environment in the parent repository. For submission onto EvalAI,
an additional package called `evalai` should be installed via `pip`.

## ```generate_groundtruth.py```

Creates the groundtruth annotations JSON file in the format stored
by the Eval AI server. Change the dataset split to `val` in order to test your validation
results. Current evaluation is supported on a randomly selected 20% subset of points from
each point cloud.

The format of the groundtruth JSON file is as follows:
```
{
    'queries': [
        {
            'horizon': '3s',
            'rays': {
                '<log_id>': {
                    '<frame_id>': List[List[List]],
                    '<frame_id>': List[List[List]]
                    ...
                },
                '<log_id>': {
                    '<frame_id>': List[List[List]],
                    '<frame_id>': List[List[List]]
                    ...
                },
                ...
            }
        }
    ]
}
```

Each '<log_id>' is a string identifier of a particular log in the Argoverse 2 Sensor suite.
Each '<frame_id>' is the current (or 0th) timestep of the 6s sequence at hand. 
3s of past is taken as input and the next 3s are to be forecasted.
Each '<frame_id>' stores the list of points in every future timestep (there are 5 future timesteps).
The number of points at each timestep can be different. Here, the List at the last level is a list of 
length 7 which stores the origin (`ox`, `oy`, `oz`), unit direction
(`dx`, `dy`, `dz`), and the expected depth along this ray (`d`).


## ```generate_query_rays.py```

Creates a set of query rays in a JSON file. Change the dataset split to `val` in order to test your
performance on the validation set. Current script generates a random 20% subset of points from each
point cloud.

The format of the JSON is as follows:
```
{
    'queries': [
        {
            'horizon': '3s',
            'rays': {
                '<log_id>': {
                    '<frame_id>': List[List[List]],
                    '<frame_id>': List[List[List]]
                    ...
                },
                '<log_id>': {
                    '<frame_id>': List[List[List]],
                    '<frame_id>': List[List[List]]
                    ...
                }
            }
        }
    ]
}
```

Each '<log_id>' is a string identifier of a particular log in the Argoverse 2 Sensor suite.
Each '<frame_id>' is the current (or 0th) timestep of the 6s sequence at hand. 
3s of past is taken as input and the next 3s are to be forecasted.
Each '<frame_id>' stores the list of points in every future timestep (there are 5 future timesteps).
The number of points at each timestep can be different. 
Here, the List at the last level is a list of length 6 which stores the origin (`ox`, `oy`, `oz`) and unit direction
(`dx`, `dy`, `dz`). When making a submission to the Eval AI server, you will replace this list of length 6, with a list
of length 1 which will store the expected depth along this ray, `d`.

## ```load_sequences.py```

Sample script to demonstrate the use of the data loader.


## ```evaluate.py```

Script to compute the metrics in the CVPR '23 paper and the leaderboard for the 4D occupancy forecasting
challenge. Usage: ```python evaluate.py --annotations /path/to/annotations --submission /path/to/submission```.




