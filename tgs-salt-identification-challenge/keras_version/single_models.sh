#!/bin/sh
#nohup sh single_models.sh > _logs/single_models.out &
#nohup sh single_models.sh poweroff > _logs/single_models.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
CLASSIFIER=unet
N_EPOCHES=200
BATCH_SIZE=32
#EXPER_NAME=single_model_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}

python single_models.py \
    --classifier ${CLASSIFIER} \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --data_augument \
    --n_samplings -1 \
    --debug

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
