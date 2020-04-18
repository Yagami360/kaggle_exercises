#!/bin/sh
#source activate pytorch11_py36
set -e

NETWORK_TYPE=resnet50
LOAD_CHECKPOINTS_PATH=checkpoints/resnet50_fc_b32_200415/model_final.hdf5
EXEP_NAME=ensemble

#-------------------
# 推論処理
#-------------------
python main_ensemble.py \
    --exper_name ${EXEP_NAME} \
    --submit_message ${EXEP_NAME} \
    --device gpu \
    --dataset_dir ../datasets \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --train_only_fc \
    --batch_size 1 \
    --n_samplings 100 \
    --n_splits 2 \
    --debug
