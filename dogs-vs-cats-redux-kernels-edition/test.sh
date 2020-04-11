#!/bin/sh
#source activate pytorch11_py36
set -e

#LOAD_CHECKPOINTS_PATH=checkpoints/resnet18_b64_200411/step_00005001.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/resnet18_b64_200411/model_final.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/resnet18_b64_norm_200411/model_final.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/resnet18_b64_norm_da_200411/step_000100001.pth
LOAD_CHECKPOINTS_PATH=checkpoints/resnet18_b64_norm_da_200411/model_final.pth

#EXEP_NAME=resnet18_b64_200411_norm_step10001
EXEP_NAME=resnet18_b64_200411_norm_da_step20001

#-------------------
# ResNet-18
#-------------------
python test.py \
    --exper_name ${EXEP_NAME} \
    --device gpu \
    --dataset_dir datasets \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --batch_size 500 \
    --debug \
    --submit
