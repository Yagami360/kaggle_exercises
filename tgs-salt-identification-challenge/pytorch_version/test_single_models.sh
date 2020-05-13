#!/bin/sh
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
MODEL_TYPE=unet
#MODEL_TYPE=unet_depth
if [ ${MODEL_TYPE} = unet ] ; then
    LOAD_CHECKPOINTS_PATH=checkpoints/single_model_unet_ep200_b32_lr0.001/model_final.hdf5
    #LOAD_CHECKPOINTS_PATH=checkpoints/single_model_unet_ep200_b32_lr0.001_da/model_final.hdf5
elif [ ${MODEL_TYPE} = unet_depth ] ; then
    LOAD_CHECKPOINTS_PATH=checkpoints/single_model_unet_depth_ep200_b32_lr0.001/model_final.hdf5
fi

python single_models.py \
    --train_mode test \
    --model_type ${MODEL_TYPE} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --n_samplings -1 \
    --debug \
    --submit

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
