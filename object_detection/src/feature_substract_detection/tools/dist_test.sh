#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

python -m torch.distributed.launch --nproc_per_node=2 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:3}

python tools/voc_eval.py eval/result.pkl $CONFIG
