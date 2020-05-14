#!/bin/sh
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
MODEL_TYPE=mgvton
LOAD_CHECKPOINTS_PATH=checkpoints/single_model_pytorch_mgvton_ep200_b32_lr0.001_da/model_ep020.pth

python single_models.py \
    --train_mode test \
    --model_type ${MODEL_TYPE} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --n_samplings 100 \
    --debug

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
