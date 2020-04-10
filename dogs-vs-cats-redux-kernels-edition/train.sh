#!/bin/sh
#source activate pytorch11_py36
#nohup sh train.sh > _logs/resnet18_b64_200410.out &
#nohup sh -c 'train.sh > _logs/resnet18_b64_200410.out && sudo poweroff' &
#nohup tensorboard --logdir tensorboard --port 6006 &
set -e

N_STEPS=10000
BATCH_SIZE=64
BATCH_SIZE_TEST=256

mkdir -p ${PWD}/_logs

#-------------------
# ResNet-18
#-------------------
EXEP_NAME=resnet18_b${BATCH_SIZE}_200410
rm -rf tensorboard/${EXEP_NAME}
rm -rf tensorboard/${EXEP_NAME}_test

python train.py \
    --device gpu \
    --exper_name ${EXEP_NAME} \
    --dataset_dir datasets \
    --n_steps ${N_STEPS} \
    --n_display_step 10 --n_display_test_step 50 \
    --debug

#sudo poweroff
#sudo shutdown -h now
