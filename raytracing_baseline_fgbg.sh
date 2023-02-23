#!/bin/bash
SPLIT="val"
BATCH_SIZE=1
NUM_WORKERS=1

python raytracing_baseline_fgbg.py --config-path baselines/nuscenes-1s.json \
    --test-split $SPLIT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --compute-chamfer-dist \
    --fg-bg "fg"

