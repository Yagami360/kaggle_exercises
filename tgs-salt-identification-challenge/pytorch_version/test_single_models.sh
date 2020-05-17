#!/bin/sh
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
MODEL_TYPE_G=unet4
#MODEL_TYPE_G=unet4bottleneck

#EXPER_NAME=single_model_pytorch_unet4_ep200_b32_lr0.001_bce1.0_enpropy1.0_l10.0_vgg0.0_advlsgan_0.0
EXPER_NAME=single_model_pytorch_unet4_ep200_b32_lr0.001_bce1.0_enpropy1.0_l10.0_vgg0.0_advlsgan_1.0
#EXPER_NAME=single_model_pytorch_unet4_da_ep200_b32_lr0.001_bce1.0_enpropy1.0_l10.0_vgg0.0_advlsgan_1.0

LOAD_CHECKPOINTS_PATH_G=checkpoints/${EXPER_NAME}/model_final.pth

python single_models.py \
    --exper_name test_${EXPER_NAME} \
    --train_mode test \
    --model_type_G ${MODEL_TYPE_G} \
    --load_checkpoints_path_G ${LOAD_CHECKPOINTS_PATH_G} \
    --n_samplings 100000 \
    --debug \
    --submit

#    --exper_name debug \
#    --exper_name test_${EXPER_NAME} \
#    --n_samplings 100 \
#    --submit

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
