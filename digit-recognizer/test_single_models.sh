#!/bin/sh
#nohup sh single_models.sh > _logs/single_models.out &
#nohup sh single_models.sh poweroff > _logs/single_models.out &
set -e

#----------------------
# model
#----------------------
LOAD_CHECKPOINTS_PATH_K1=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k1_final.hdf5
LOAD_CHECKPOINTS_PATH_K2=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k2_final.hdf5
LOAD_CHECKPOINTS_PATH_K3=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k3_final.hdf5
LOAD_CHECKPOINTS_PATH_K4=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k4_final.hdf5

CLASSIFIER=resnet50
N_EPOCHES=100
BATCH_SIZE=64
N_SPLITS=4
EXPER_NAME=test_single_model_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}_k${N_SPLITS}_da

python single_models.py \
    --exper_name ${EXPER_NAME} \
    --train_mode test \
    --classifier ${CLASSIFIER} \
    --n_splits ${N_SPLITS} \
    --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_K1} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_K2} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_K3} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_K4} \
    --debug \
    --submit
