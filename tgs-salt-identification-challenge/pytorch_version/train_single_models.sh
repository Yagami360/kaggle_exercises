#!/bin/sh
#nohup sh train_single_models.sh > _logs/train_single_models_8.out &
#nohup sh train_single_models.sh poweroff > _logs/train_single_models_9.out &
#nohup sh train_single_models.sh > _logs/train_single_models_10.out &
#nohup sh train_single_models.sh poweroff > _logs/train_single_models_11.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
#MODEL_TYPE_G=unet4
#MODEL_TYPE_G=unet4bottleneck
MODEL_TYPE_G=unet4resnet34
#MODEL_TYPE_G=mgvton
#MODEL_TYPE_G=ganimation

MODEL_TYPE_D=patchgan
#MODEL_TYPE_D=ganimation

N_EPOCHES=200
BATCH_SIZE=32
EXPER_NAME=debug
#rm -rf tensorboard/${EXPER_NAME}

python single_models.py \
    --train_mode train \
    --model_type_G ${MODEL_TYPE_G} --model_type_D ${MODEL_TYPE_D} \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --lambda_bce 0.0 --lambda_enpropy 0.0 --lambda_lovasz_softmax 1.0 --lambda_l1 0.1 --lambda_adv 0.0 --lambda_cond 0.0 \
    --debug \
    --submit

#    --data_augument \
#    --exper_name ${EXPER_NAME} \

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
