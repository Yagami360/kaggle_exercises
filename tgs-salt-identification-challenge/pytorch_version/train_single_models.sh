#!/bin/sh
#nohup sh single_models.sh > _logs/single_models.out &
#nohup sh single_models.sh poweroff > _logs/single_models.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
MODEL_TYPE=unet
N_EPOCHES=200
BATCH_SIZE=32
EXPER_NAME=debug
rm -rf tensorboard/${EXPER_NAME}

python single_models.py \
    --exper_name ${EXPER_NAME} \
    --train_mode train \
    --model_type ${MODEL_TYPE} \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --debug

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
