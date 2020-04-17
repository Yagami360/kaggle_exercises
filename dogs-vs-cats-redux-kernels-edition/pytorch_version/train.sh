#!/bin/sh
#source activate pytorch11_py36
#nohup sh train.sh > _logs/resnet50_b64_norm_da_200411.out &
#nohup sh train.sh > _logs/resnet50_fc_b64_norm_da_200411.out &
#nohup tensorboard --logdir tensorboard --port 6006 &
set -e

N_STEPS=10000
BATCH_SIZE=32
mkdir -p ${PWD}/_logs

#NETWORK_TYPE=my_resnet18
#NETWORK_TYPE=resnet18
NETWORK_TYPE=resnet50

#-------------------
# 学習処理
#-------------------
EXEP_NAME=debug
#EXEP_NAME=${NETWORK_TYPE}_b${BATCH_SIZE}_200411
#EXEP_NAME=${NETWORK_TYPE}_b${BATCH_SIZE}_norm_200411
#EXEP_NAME=${NETWORK_TYPE}_b${BATCH_SIZE}_norm_da_200411
EXEP_NAME=${NETWORK_TYPE}_fc_b${BATCH_SIZE}_norm_da_200411
rm -rf tensorboard/${EXEP_NAME}
rm -rf tensorboard/${EXEP_NAME}_test

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir ../datasets \
    --network_type ${NETWORK_TYPE} \
    --pretrained --train_only_fc \
    --n_steps ${N_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --n_display_step 50 \
    --enable_da \
    --debug

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
