#!/bin/sh
#source activate pytorch11_py36
#nohup sh train.sh > _logs/resnet50_kfold3_b32_norm_da_200411.out &
#nohup tensorboard --logdir tensorboard --port 6006 &
set -e

N_SPLITS=3
N_STEPS=10
BATCH_SIZE=32
BATCH_SIZE_TEST=256
mkdir -p ${PWD}/_logs

#NETWORK_TYPE=my_resnet18
#NETWORK_TYPE=resnet18
NETWORK_TYPE=resnet50

#-------------------
# 学習処理
#-------------------
EXEP_NAME=debug
#EXEP_NAME=${NETWORK_TYPE}_${BATCH_SIZE}_norm_da_200411
rm -rf tensorboard/${EXEP_NAME}_kfold*
rm -rf tensorboard/${EXEP_NAME}_kfold*_test

python train_kfold.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir datasets \
    --network_type ${NETWORK_TYPE} \
    --pretrained \
    --n_splits ${N_SPLITS} \
    --n_steps ${N_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --n_display_step 50 --n_display_test_step 500 \
    --enable_da \
    --debug

#sudo poweroff
#sudo shutdown -h now
