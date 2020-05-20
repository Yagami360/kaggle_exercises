#!/bin/sh
#nohup sh train_single_models.sh > _logs/train_single_models_1.out &
#nohup sh train_single_models.sh poweroff > _logs/train_single_models_2.out &
#nohup sh train_single_models.sh > _logs/train_single_models_5.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
MODEL_TYPE_G=unet4
#MODEL_TYPE_G=unet4bottleneck
#MODEL_TYPE_G=mgvton
#MODEL_TYPE_G=ganimation

MODEL_TYPE_D=patchgan
#MODEL_TYPE_D=ganimation

N_EPOCHES=200
BATCH_SIZE=32
EXPER_NAME=debug
#rm -rf tensorboard/${EXPER_NAME}

python single_models.py \
    --exper_name ${EXPER_NAME} \
    --train_mode train \
    --model_type_G ${MODEL_TYPE_G} --model_type_D ${MODEL_TYPE_D} \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --lambda_bce 1.0 --lambda_enpropy 1.0 --lambda_lovasz_softmax 1.0  --lambda_l1 0.0 --lambda_adv 0.0 --lambda_cond 0.0 \
    --debug

#    --data_augument \

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
