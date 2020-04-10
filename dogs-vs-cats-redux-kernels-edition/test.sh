#!/bin/sh
#source activate pytorch11_py36
set -e

#-------------------
# ResNet-18
#-------------------
python test.py \
    --device gpu \
    --dataset_dir datasets \
    --load_checkpoints_path checkpoints/resnet18_b64_200410/model_final.pth \
    --batch_size 64 \
    --n_samplings 5 \
    --debug

