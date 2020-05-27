#!/bin/sh
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
MODEL_TYPE_G=unet4
#MODEL_TYPE_G=unet4resnet34

EXPER_NAME=debug
LOAD_CHECKPOINTS_PATH_G=checkpoints/${EXPER_NAME}/model_final.pth

python single_models.py \
    --exper_name test_${EXPER_NAME} \
    --train_mode test \
    --model_type_G ${MODEL_TYPE_G} \
    --load_checkpoints_path_G ${LOAD_CHECKPOINTS_PATH_G} \
    --n_samplings 10 \
    --debug

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
