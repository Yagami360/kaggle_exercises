#!/bin/sh
#source activate pytorch11_py36
set -e

NETWORK_TYPE1=resnet50
LOAD_CHECKPOINTS_PATH1=checkpoints/resnet50_fc_b32_norm_da_200411/model_final.pth
EXEP_NAME=ensemble_resnet_and_noneNN

#-------------------
# 推論処理
#-------------------
python main_ensemble.py \
    --exper_name ${EXEP_NAME} \
    --submit_message ${EXEP_NAME} \
    --device gpu \
    --dataset_dir datasets \
    --network_type ${NETWORK_TYPE1} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH1} \
    --batch_size 1 \
    --n_samplings 100 \
    --debug
