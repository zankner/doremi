#!/bin/bash

#
# Sample baseline model run of a 280M model with the same number of non-embedding parameters as the 280M model in the DoReMi paper. Not the same as DoReMi paper since the paper uses 256k vocab size.
#


# load global parameters
source constants.sh
pip install -e .

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE
export WANDB_DIR=${CACHE}/wandb
export WANDB_PROJECT="doremi-ablate-repro"

PREPROCESSED_DATA=${PREPROCESSED_PILE_DIR}
PREPROCESSED_CACHE=${CACHE}/preprocessed_cache/perdomain_pile_preprocessed

if [ ! -d "${PREPROCESSED_CACHE}" ]; then
    mkdir -p ${CACHE}/preprocessed_cache
    cp -r ${PREPROCESSED_DATA} ${PREPROCESSED_CACHE}
fi

arg=${1:-""} # set to eval to run eval

if [[ "${arg}" == "eval" ]]; then
    ADDITIONAL_ARGS="--evaluation_strategy steps --per_device_eval_batch_size 32 --do_train false --remove_unused_columns=False"
else
    ADDITIONAL_ARGS=""
fi


NAME=complete_replicate_pile_neox_final_doremi_280M_50kvocab
accelerate launch \
    --config_file accelerate_config.yml \
    --multi_gpu \
    --num_machines 4 \
    --num_processes 32 \
    --machine_rank $NODE_RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    doremi/train.py \
    --dataset_name pile \
    --model_name mpt-280M \
    --tokenizer_name EleutherAI/gpt-neox-20b \
    --do_train \
    --cache_dir ${CACHE} \
    --domain_config_path configs/pile_doremi_noex_50kvocab.json \
    --output_dir ${MODEL_OUTPUT_DIR}/${NAME} \
    --max_token_length 1024 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --max_steps 200000 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --learning_rate 1e-3 \
    --lr_end 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-8 \
    --lr_scheduler_name linear_warmup_cosine \
    --warmup_ratio 0.06 \
    --run_name ${NAME} \
    --seed 17 \
    --logging_strategy steps \
    --logging_steps 100 \
    --logging_first_step \
    --report_to wandb \
    --optim adafactor \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --bf16 \
    ${ADDITIONAL_ARGS}
