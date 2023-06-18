#!/bin/bash

# This is a sample of the constants file. Please write any env variables here
# and rename the file constants.sh

CACHE=/tmp/doremi/cache
DOREMI_DIR=/mnt/workdisk/zack/doremi
MODEL_OUTPUT_DIR=/tmp/doremi/models
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
source ${DOREMI_DIR}/env/bin/activate  # if you installed doremi in venv
