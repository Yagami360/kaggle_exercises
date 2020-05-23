#!/bin/sh
#nohup sh train_single_models.sh > _logs/train_single_models_1.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
MODEL_TYPE_G=unet4
MODEL_TYPE_D=patchgan

N_EPOCHES=2
BATCH_SIZE=32
EXPER_NAME=debug
#rm -rf tensorboard/${EXPER_NAME}

python single_models.py \
    --exper_name ${EXPER_NAME} \
    --train_mode train \
    --model_type_G ${MODEL_TYPE_G} --model_type_D ${MODEL_TYPE_D} \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --image_height 256 --image_width 192 --n_channels 3 \
    --debug

#    --data_augument \


if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
