#!/bin/sh
#source activate pytorch11_py36
#nohup sh train.sh > _logs/resnet18_b64_norm_da_200411.out &
#nohup tensorboard --logdir tensorboard --port 6006 &
set -e

N_STEPS=1000
BATCH_SIZE=64
BATCH_SIZE_TEST=256
mkdir -p ${PWD}/_logs

NETWORK_TYPE=resnet50

#-------------------
# ResNet-18
#-------------------
EXEP_NAME=debug
#EXEP_NAME=${NETWORK_TYPE}_b${BATCH_SIZE}_200411
#EXEP_NAME=${NETWORK_TYPE}_b${BATCH_SIZE}_norm_200411
#EXEP_NAME=${NETWORK_TYPE}_b${BATCH_SIZE}_norm_da_200411
rm -rf tensorboard/${EXEP_NAME}
rm -rf tensorboard/${EXEP_NAME}_test

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir datasets \
    --n_steps ${N_STEPS} \
    --network_type ${NETWORK_TYPE} \
    --pretrained \
    --batch_size ${BATCH_SIZE} \
    --n_display_step 50 --n_display_test_step 50 \
    --enable_da \
    --debug

#sudo poweroff
#sudo shutdown -h now
