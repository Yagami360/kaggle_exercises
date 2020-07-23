#!/bin/sh
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
#MODEL_TYPE_G=unet4
#MODEL_TYPE_G=unet4_resnet
#MODEL_TYPE_G=unet_fgvc6
MODEL_TYPE_G=deeplab_v3

EXPER_NAME=debug
LOAD_CHECKPOINTS_PATH_G=checkpoints/${EXPER_NAME}/model_final.pth
rm -r results/${EXPER_NAME}
rm -r tensorboard/${EXPER_NAME}_test

python single_models.py \
    --exper_name ${EXPER_NAME} \
    --train_mode test \
    --model_type_G ${MODEL_TYPE_G} \
    --load_checkpoints_path_G ${LOAD_CHECKPOINTS_PATH_G} \
    --n_samplings 1 \
    --debug

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
