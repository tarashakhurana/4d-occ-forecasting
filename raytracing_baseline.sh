#!/bin/bash
SPLIT="test"
BATCH_SIZE=4
NUM_WORKERS=4

python raytracing_baseline.py --config-path baselines/nuscenes-1s.json \
    --test-split $SPLIT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --compute-chamfer-dist

