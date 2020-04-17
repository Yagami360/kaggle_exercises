#!/bin/sh
#source activate tensorflow_p36
#nohup sh train.sh > _logs/resnet50_fc_b32__200415.out &
#nohup sh train.sh poweroff > _logs/resnet50_fc_b32__200415.out &
set -e
mkdir -p ${PWD}/_logs

N_STEPS=5000
BATCH_SIZE=32
NETWORK_TYPE=resnet50

#-------------------
# 学習処理
#-------------------
EXEP_NAME=debug
EXEP_NAME=${NETWORK_TYPE}_fc_b${BATCH_SIZE}_200415
rm -rf tensorboard/${EXEP_NAME}
rm -rf tensorboard/${EXEP_NAME}_test

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir ../datasets \
    --use_tensorboard \
    --network_type ${NETWORK_TYPE} \
    --pretrained --train_only_fc \
    --n_steps ${N_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --enable_datagen \
    --n_display_step 50 \
    --debug

#    --enable_da \

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
