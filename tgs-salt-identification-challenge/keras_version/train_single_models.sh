#!/bin/sh
#nohup sh single_models.sh > _logs/single_models.out &
#nohup sh single_models.sh poweroff > _logs/single_models.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
#MODEL_TYPE=unet
MODEL_TYPE=unet_depth
N_EPOCHES=200
BATCH_SIZE=32
DATA_AUGUMENT_TYPE=da1
mkdir -p results/${EXPER_NAME}

python single_models.py \
    --train_mode train \
    --classifier ${MODEL_TYPE} \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --data_augument_type ${DATA_AUGUMENT_TYPE} \
    --n_samplings 100 \
    --debug

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
