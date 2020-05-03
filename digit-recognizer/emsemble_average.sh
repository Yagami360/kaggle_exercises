#!/bin/sh
#nohup sh emsemble_average.sh > _logs/emsemble_average.out &
#nohup sh emsemble_average.sh poweroff > _logs/emsemble_average.out &
set -e
mkdir -p _logs

#----------------------
# model
#----------------------
LOAD_CHECKPOINTS_PATH_C1_K1=checkpoints/single_model_catboost_iter1000_lr0.01_k4_da/model_k1_final.json
LOAD_CHECKPOINTS_PATH_C1_K2=checkpoints/single_model_catboost_iter1000_lr0.01_k4_da/model_k2_final.json
LOAD_CHECKPOINTS_PATH_C1_K3=checkpoints/single_model_catboost_iter1000_lr0.01_k4_da/model_k3_final.json
LOAD_CHECKPOINTS_PATH_C1_K4=checkpoints/single_model_catboost_iter1000_lr0.01_k4_da/model_k4_final.json

LOAD_CHECKPOINTS_PATH_C2_K1=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k1_final.hdf5
LOAD_CHECKPOINTS_PATH_C2_K2=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k2_final.hdf5
LOAD_CHECKPOINTS_PATH_C2_K3=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k3_final.hdf5
LOAD_CHECKPOINTS_PATH_C2_K4=checkpoints/single_model_k4_da_resnet50_ep100_b64/model_k4_final.hdf5

LOAD_CHECKPOINTS_PATH_C3_K1=checkpoints/single_model_k4_da_pretrained_resnet50_ep100_b64/model_k1_final.hdf5
LOAD_CHECKPOINTS_PATH_C3_K2=checkpoints/single_model_k4_da_pretrained_resnet50_ep100_b64/model_k2_final.hdf5
LOAD_CHECKPOINTS_PATH_C3_K3=checkpoints/single_model_k4_da_pretrained_resnet50_ep100_b64/model_k3_final.hdf5
LOAD_CHECKPOINTS_PATH_C3_K4=checkpoints/single_model_k4_da_pretrained_resnet50_ep100_b64/model_k4_final.hdf5

python emsemble_average.py \
    --train_modes test --train_modes test --train_modes test \
    --vote_method majority_vote \
    --classifiers catboost --classifiers resnet50 --classifiers pretrained_resnet50 \
    --weights 0.2 --weights 0.4 --weights 0.4 \
    --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K1} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K2} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K3} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C1_K4} \
    --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K1} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K2} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K3} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C2_K4} \
    --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K1} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K2} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K3} --load_checkpoints_paths ${LOAD_CHECKPOINTS_PATH_C3_K4} \
    --n_splits 4 \
    --debug \
    --submit

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
