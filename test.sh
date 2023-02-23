#!/bin/bash
SPLIT="test"
BATCH_SIZE=2
NUM_WORKERS=2

python test.py --model-dir models/kitti/1s_forecasting \
    --test-epoch 14 \
    --test-split $SPLIT \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --compute-chamfer-dist
