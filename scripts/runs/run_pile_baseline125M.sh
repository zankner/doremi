#!/bin/bash

#
# Sample baseline model run of a 280M model with the same number of non-embedding parameters as the 280M model in the DoReMi paper. Not the same as DoReMi paper since the paper uses 256k vocab size.
#


# load global parameters
source constants.sh
# pip install --use-pep51 -e .

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE
export WANDB_DIR=${CACHE}/wandb
export WANDB_PROJECT=official-doremi

PREPROCESSED_DATA=${PREPROCESSED_PILE_DIR}
PREPROCESSED_CACHE=${CACHE}/preprocessed_cache/perdomain_pile_preprocessed

if [ ! -d "${PREPROCESSED_CACHE}" ]; then
    mkdir -p ${CACHE}/preprocessed_cache
    cp -r ${PREPROCESSED_DATA} ${PREPROCESSED_CACHE}
fi

# USING ADAFACTOR BUT FOUNDRY USES LION
# Set max steps as 100_000 and save steps as 5_000

NAME=pile_baseline_125M
accelerate launch \
    --config_file accelerate_config.yml \
    --num_processes 8 \
    --multi_gpu \
    --num_machines 1 \
    --main_process_port 60200 \
    doremi/train.py \
    --dataset_name pile \
    --tokenizer_name EleutherAI/gpt-neox-20b \
    --do_train \
    --cache_dir ${CACHE} \
    --dataset_dir ${PREPROCESSED_CACHE} \
    --domain_config_path configs/mpt_pile_baseline_50kvocab.json \
    --output_dir ${MODEL_OUTPUT_DIR}/${NAME} \
    --max_token_length 2048 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 8 \
    --max_steps 100 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50 \
    --learning_rate 2.0e-4 \
    --lr_end 0 \
    --weight_decay 0.0006 \
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
    --adam_beta2 0.95 \
    --bf16 \
