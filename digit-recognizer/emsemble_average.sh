#!/bin/sh
#nohup sh emsemble_average.sh > _logs/emsemble_average.out &
#nohup sh emsemble_average.sh poweroff > _logs/emsemble_average.out &
set -e
mkdir -p _logs

#----------------------
# model
#----------------------
LOAD_CHECKPOINTS_PATH_C1_K1=dummy
LOAD_CHECKPOINTS_PATH_C1_K2=dummy
LOAD_CHECKPOINTS_PATH_C1_K3=dummy
LOAD_CHECKPOINTS_PATH_C1_K4=dummy

LOAD_CHECKPOINTS_PATH_C2_K1=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k1_final.hdf5
LOAD_CHECKPOINTS_PATH_C2_K2=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k2_final.hdf5
LOAD_CHECKPOINTS_PATH_C2_K3=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k3_final.hdf5
LOAD_CHECKPOINTS_PATH_C2_K4=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k4_final.hdf5

LOAD_CHECKPOINTS_PATH_C3_K1=dummy
LOAD_CHECKPOINTS_PATH_C3_K2=dummy
LOAD_CHECKPOINTS_PATH_C3_K3=dummy
LOAD_CHECKPOINTS_PATH_C3_K4=dummy

python emsemble_average.py \
    --train_modes train --train_modes eval --train_modes eval \
    --vote_method majority_vote \
    --classifiers catboost --classifiers resnet50 --classifiers pretrained_resnet50 \
    --weights 0.2 --weights 0.4 --weights 0.4 \
    --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K1} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K2} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K3} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K4} \
    --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K1} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K2} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K3} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K4} \
    --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K1} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K2} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K3} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K4} \
    --n_splits 4 \
    --debug

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
