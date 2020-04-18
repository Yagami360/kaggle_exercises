#!/bin/sh
#source activate pytorch11_py36
set -e

NETWORK_TYPE=resnet50
LOAD_CHECKPOINTS_PATH=checkpoints/${NETWORK_TYPE}_fc_b32_200415/model_final.hdf5
EXEP_NAME=${NETWORK_TYPE}_fc_b32_200415

#-------------------
# 推論処理
#-------------------
python test.py \
    --exper_name ${EXEP_NAME} \
    --device gpu \
    --dataset_dir ../datasets \
    --network_type ${NETWORK_TYPE} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --train_only_fc \
    --batch_size 1 \
    --n_samplings 100000 \
    --enable_datagen \
    --debug \
    --submit
