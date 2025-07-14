args=$@
for arg in $args; do
    eval "$arg"
done

echo "model:            ${model:=}"
echo "tokenizer:        ${tokenizer:=}"
echo "project:          ${project:=}"
echo "type:             ${type:=}"
echo "data:             ${data:=}"
echo "name:             ${name:=}"
echo "cache:            ${cache:=}"
echo "seed:             ${seed:=42}"
echo "context:          ${context:=}"
echo "steps:            ${steps:=}"
echo "save:             ${save:=}" 
echo "limit:            ${limit:=}"
echo "preprocessing:    ${preprocessing:=}"
echo "workers:          ${workers:=}"
echo "logging:          ${logging:=}"
echo "config:           ${config:=configs/deepspeed.yaml}"

echo "lr:               ${lr:=}"
echo "scheduler:        ${scheduler:=}"
echo "epochs:           ${epochs:=}"
echo "optim:            ${optim:=}"
echo "decay:            ${decay:=}"
echo "beta1:            ${beta1:=}"
echo "beta2:            ${beta2:=}"
echo "norm:             ${norm:=}"
echo "batch:            ${batch:=}"
echo "update:           ${update:=}"
echo "warmup:           ${warmup:=}"
echo "path:             ${path:=}"
echo "checkpoint:       ${checkpoint:=}"
echo "node:             ${node:=}"
echo "rank:             ${rank:=}"
echo "ip:               ${ip:=}"
echo "port:             ${port:=}"
echo "nodes:            ${nodes:=1}"
echo "soft_loss:        ${soft_loss:=}"
echo "soft_loss_decay:  ${soft_loss_decay:=}"
echo "align_loss:       ${align_loss:=}"
echo "loss_func:        ${loss_func:=}"
echo "min_lr_rate:      ${min_lr_rate:=}"

params="--model_name_or_path $model \
    --tokenizer $tokenizer \
    --use_fast_tokenizer \
    --do_train \
    --dataset $data \
    --context_length $context \
    --streaming \
    --preprocessing_num_workers $preprocessing \
    --dataloader_num_workers $workers \
    --dataloader_prefetch_factor 2 \
    --ignore_data_skip \
    --output_dir $path \
    --overwrite_output_dir \
    --logging_steps $logging \
    --include_num_input_tokens_seen \
    --save_steps $save \
    --save_total_limit $limit \
    --learning_rate $lr \
    --lr_scheduler_type $scheduler \
    --warmup_steps $warmup \
    --optim $optim \
    --weight_decay $decay \
    --adam_beta1=$beta1 \
    --adam_beta2=$beta2 \
    --max_grad_norm $norm \
    --num_train_epochs $epochs \
    --per_device_train_batch_size $batch \
    --gradient_accumulation_steps $update \
    --seed $seed \
    --logging_steps $logging \
    --freeze \
    --soft_loss $soft_loss \
    --soft_loss_decay $soft_loss_decay \
    --align_loss $align_loss \
    --loss_func $loss_func \
    --min_lr_rate $min_lr_rate \
    --bf16"

if [ $steps -gt 0 ]; then
    params+=" --max_steps $steps"
fi

if [ "$name" != "" ]; then
  params+=" --dataset_name $name"
fi
if [ "$cache" != "" ]; then
  params+=" --cache_dir $cache"
fi
if [ "$checkpoint" != "" ]; then
  params+=" --resume_from_checkpoint $checkpoint"
fi
if [ "$WANDB_DISABLED" != "true" ]; then
  params+=" --report_to wandb \
  --run_name $type.$(basename $path)"
else
  params+=" --report_to none"
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Launching training..."
accelerate_params=""
if [ "$rank" != "" ]; then
  accelerate_params+=" --machine_rank $rank  \
    --num_processes $((nodes * $NUM_GPUS)) \
    --num_machines $nodes \
    --main_process_ip $ip \
    --main_process_port $port \
    --same_network"
fi

if [[ $config == *"deepspeed"* ]]; then
cat <<EOF > "configs/ds_config.json"
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": false,
    "contiguous_gradients": true
  }
}
EOF
cat <<EOF > $config
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: configs/ds_config.json
  zero3_init_flag: true
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: $NUM_GPUS
use_cpu: false
EOF
fi
if [[ $config == *"fsdp"* ]]; then
cat <<EOF > $config
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: HYBRID_SHARD_ZERO2
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: $nodes
num_processes: $((nodes * $NUM_GPUS))
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

cat $config

set -x
mkdir -p $path

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [ "$date" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi
export WANDB_RESUME=allow
export WANDB_NAME="$type.$(basename $path)"
export WANDB_PROJECT=$project
export WANDB_RUN_ID="$WANDB_NAME-$date"
accelerate launch $accelerate_params --config_file $config run_distill.py $params 
