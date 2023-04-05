# Argoverse 2 4D Occupancy Forecasting Evaluation Kit

Official helper scripts for the 2023 Argoverse 2.0 4D Occupancy Forecasting Challenge
at the CVPR 2023 Workshop on Autonomous Driving.

## Installation

This evaluation kit only requires the environment in the parent repository. For submission onto EvalAI,
an additional package called `evalai` should be install via `pip`.

## ```generate_groundtruth.py```

Creates the groundtruth annotations JSON file in the format stored
by the Eval AI server. Change the dataset split to `val` in order to test your validation
results. Current evaluation is supported on a randomly selected 20% subset of points from
each point cloud.

The format of the groundtruth JSON file is as follows:
```
```

## ```generate_query_rays.py```

Creates a set of query rays in a JSON file. Change the dataset split to `val` in order to test your
performance on the validation set. Current script generates a random 10% subset of points from each
point cloud.

The format of the JSON is as follows:
```
```

## ```load_sequences.py```

Sample script to demonstrate the use of the data loader.


## ```evaluate.py```

Script to compute the metrics in the CVPR '23 paper and the leaderboard for the 4D occupancy forecasting
challenge. Usage: ```python evaluate.py <groundtruth file path> <submission file path>```.




