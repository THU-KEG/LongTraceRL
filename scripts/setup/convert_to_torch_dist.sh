#!/bin/bash
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
HF_MODEL_PATH=$PROJECT_ROOT/models/hf_models/LongTraceRL-4B
OUTPUT_PATH=$PROJECT_ROOT/models/torch_dist_models/LongTraceRL-4B

SLIME_DIR=slime
MEGATRON_LM_DIR=/root/Megatron-LM

cd $SLIME_DIR

# Get MODEL_ARGS
source scripts/models/qwen3-4B-Instruct-2507.sh

PYTHONPATH=$MEGATRON_LM_DIR \
python3 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint $HF_MODEL_PATH \
    --save $OUTPUT_PATH
