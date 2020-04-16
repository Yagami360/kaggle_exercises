#!/bin/sh
#source activate pytorch11_py36
set -e

NETWORK_TYPE=my_resnet18
#NETWORK_TYPE=resnet18
#NETWORK_TYPE=resnet50

#LOAD_CHECKPOINTS_PATH=checkpoints/my_resnet18_b64_200411/step_00005001.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/my_resnet18_b64_200411/model_final.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/my_resnet18_b64_norm_200411/model_final.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/my_resnet18_b64_norm_da_200411/step_000100001.pth
LOAD_CHECKPOINTS_PATH=checkpoints/my_resnet18_b64_norm_da_200411/model_final.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/${NETWORK_TYPE}_b32_norm_da_200411/model_final.pth
#LOAD_CHECKPOINTS_PATH=checkpoints/${NETWORK_TYPE}_fc_b32_norm_da_200411/step_00001001.pth

EXEP_NAME=${NETWORK_TYPE}_b64_200411_norm_da
#EXEP_NAME=${NETWORK_TYPE}_b32_200411_norm_da
#EXEP_NAME=${NETWORK_TYPE}_fc_b32_200411_norm_da

#-------------------
# 推論処理
#-------------------
python test.py \
    --exper_name ${EXEP_NAME} \
    --submit_message ${EXEP_NAME} \
    --device gpu \
    --dataset_dir ../datasets \
    --network_type ${NETWORK_TYPE} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --batch_size 1 \
    --n_samplings 100000 \
    --debug \
    --submit