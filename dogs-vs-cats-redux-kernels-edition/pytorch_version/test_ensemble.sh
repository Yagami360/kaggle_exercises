#!/bin/sh
#source activate pytorch11_py36
set -e

NETWORK_TYPE1=my_resnet18
NETWORK_TYPE2=resnet50
NETWORK_TYPE3=resnet50

LOAD_CHECKPOINTS_PATH1=checkpoints/my_resnet18_b64_norm_da_200411/model_final.pth
LOAD_CHECKPOINTS_PATH2=checkpoints/resnet50_b32_norm_da_200411/model_final.pth
LOAD_CHECKPOINTS_PATH3=checkpoints/resnet50_fc_b32_norm_da_200411/model_final.pth

EXEP_NAME=ensemble_resnet

#-------------------
# 推論処理
#-------------------
python test_ensemble.py \
    --exper_name ${EXEP_NAME} \
    --device gpu \
    --dataset_dir ../datasets \
    --network_type ${NETWORK_TYPE1} --network_type ${NETWORK_TYPE2} --network_type ${NETWORK_TYPE3} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH1} --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH2} --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH3} \
    --batch_size 1 \
    --n_samplings 100000 \
    --debug \
    --submit
    