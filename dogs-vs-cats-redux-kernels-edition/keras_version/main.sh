#!/bin/sh
#source activate pytorch11_py36
set -e

N_STEPS=100
BATCH_SIZE=32
mkdir -p ${PWD}/_logs

NETWORK_TYPE=resnet50

#-------------------
# 学習処理
#-------------------
EXEP_NAME=debug
rm -rf tensorboard/${EXEP_NAME}
rm -rf tensorboard/${EXEP_NAME}_test

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir datasets \
    --network_type ${NETWORK_TYPE} \
    --pretrained --train_only_fc \
    --n_steps ${N_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --n_display_step 50 \
    --enable_da \
    --debug

