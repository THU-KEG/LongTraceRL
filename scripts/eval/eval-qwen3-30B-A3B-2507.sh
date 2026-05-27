#!/bin/bash

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SLIME_DIR=${PROJECT_ROOT}/slime
cd ${SLIME_DIR}

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# model parallelism
TP_SIZE=4
PP_SIZE=2
CP_SIZE=4
EP_SIZE=16
ETP_SIZE=1

# rollout engine
ROLLOUT_TP_SIZE=8
ROLLOUT_MEM_UTILIZATION=0.7

# context length
MAX_CONTEXT_LEN=160000
MAX_GEN_LEN=32000
MAX_EVAL_CONTEXT_LEN=160000
MAX_EVAL_GEN_LEN=32000

# exp
EXP_TAG=eval-qwen3-30B-A3B-2507
REF_MODEL_PATH=$PROJECT_ROOT/models/torch_dist_models/LongTraceRL-30B
HF_MODEL_PATH=$PROJECT_ROOT/models/hf_models/LongTraceRL-30B
CKPT_DIR=$PROJECT_ROOT/outputs/${EXP_TAG}

# Qwen3-30B-A3B (MoE)
MODEL_ARGS=(
   --max-position-embeddings ${MAX_CONTEXT_LEN}
   --seq-length ${MAX_CONTEXT_LEN}
   --position-embedding-type rope
   --rotary-percent 1.0
   --rotary-base 10000000
   --swiglu
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --num-layers 48
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 4
   --kv-channels 128
   --qk-layernorm
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --vocab-size 151936
   --make-vocab-size-divisible-by 32
   --untie-embeddings-and-output-weights
   --attention-softmax-in-fp32
   --attention-backend flash
   # moe
   --num-experts 128
   --moe-router-topk 8
   --moe-ffn-hidden-size 768
   --moe-layer-freq "'([1]*48)'"
   --moe-router-score-function softmax
   --moe-router-dtype fp32
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-permute-fusion
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${REF_MODEL_PATH}
   --save-interval 20
   --save ${CKPT_DIR}
   --load ${CKPT_DIR}
   --keep-optim-recent 1
   --ckpt-format torch_dist
   --tokenizer-type HuggingFaceTokenizer
   --tokenizer-model ${HF_MODEL_PATH}
   --no-load-rng
)

ROLLOUT_ARGS=(
   --custom-generate-function-path slime.rollout.rollout_skip_illform.generate
   --prompt-data $PROJECT_ROOT/data/train/data.jsonl
   # --apply-chat-template
   --source-key source
   --input-key input_messages
   --label-key label
   --source-config-path configs/source_config_qwen3_rubric.json
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --global-batch-size 128
   --num-rollout 200
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-shuffle
   --rollout-temperature 1.0
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --sglang-server-concurrency 128
   --rollout-stop-token-ids 151643 151645
   --rubric-reward-ratio 0.3
   --normalize-rubric-reward
   --rubric-only-positive
   # --rubric-use-reasoning-content
   --rubric-use-content
   --only-eval
)

EVAL_ARGS=(
   # --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aa-lcr $PROJECT_ROOT/data/test/aa_lcr.jsonl
   --eval-input-key input_messages
   --eval-label-key label
   --n-samples-per-eval-prompt 4
   --eval-max-context-len ${MAX_EVAL_CONTEXT_LEN}
   --eval-max-response-len ${MAX_EVAL_GEN_LEN}
   --eval-temperature 0.6
   --eval-top-p 0.95
   --eval-top-k 20
   --eval-results-save-dir ${CKPT_DIR}/eval_results
)

DISTRIBUTED_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size ${ETP_SIZE}
   --sequence-parallel
)

PERF_ARGS=(
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --calculate-per-token-loss
   --use-tis
   --custom-tis-function-path slime.backends.megatron_utils.loss.icepop_function
   --tis-clip-low 0.5
   --tis-clip 2.0
)

OPTIMIZER_ARGS=(
   --lr 2e-6
   --lr-warmup-iters 0
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --override-opt_param-scheduler
)

WANDB_ARGS=(
   --use-wandb
   --wandb-host https://wandb.glm.ai/
   --wandb-key ${WANDB_API_KEY}
   --wandb-team glm-zero
   --wandb-project ${WANDB_PROJECT}
   --wandb-group ${EXP_TAG}
   --disable-wandb-random-suffix
   --wandb-always-use-train-step
)


# network
export MASTER_ADDR=${MLP_WORKER_0_HOST}
export MASTER_PORT=${MLP_WORKER_0_PORT}
export GLOO_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}
export NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME}
export no_proxy=localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}

# launch the master node of ray in container
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
   if [[ "$WORKER_IP" == "$MLP_WORKER_0_HOST" ]]; then
      continue
   fi
   echo "Starting Ray worker on ${WORKER_IP}"
   ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address=${MASTER_ADDR}:6379 --num-gpus 8 --node-ip-address ${WORKER_IP} --disable-usage-stats" &
done
wait

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
    "GLOO_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${MLP_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MLP_WORKER_0_HOST}",
    "MASTER_PORT": "${MLP_WORKER_0_PORT}",
    "PYTHONPATH": "/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "0",
    "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
    "NCCL_NET_GDR_LEVEL": "2",
    "NCCL_IB_QPS_PER_CONNECTION": "2",
    "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
    "NCCL_IB_TC": "160",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_NET_GDR_LEVEL": "4",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_TIMEOUT": "32",
    "NCCL_IB_QPS_PER_CONNECTION": "8",
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NCCL_PXN_DISABLE": "0",
    "NCCL_MIN_CTAS": "4",
    "OMPI_MCA_pml": "ob1",
    "OMPI_MCA_btl": "^openib",
    "OMPI_MCA_routed": "direct",
    "OMPI_MCA_routed_radix": "1024",
    "OMPI_MCA_plm_rsh_no_tree_spawn": "1",
    "OMPI_MCA_oob_tcp_if_include": "${MLP_SOCKET_IFNAME}",
    "OMPI_MCA_btl_tcp_if_include": "${MLP_SOCKET_IFNAME}"
  }
}
EOF
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${MLP_WORKER_NUM} \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus $(( ${MLP_WORKER_NUM} * 8 )) \
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE} \
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION} \
   --sglang-ep-size 8 \
   --sglang-router-request-timeout-secs 36000 \
   --sglang-router-balance-abs-threshold 0 \
   --offload \
   --colocate \
   --no-check-for-nan-in-loss-and-grad \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]}
