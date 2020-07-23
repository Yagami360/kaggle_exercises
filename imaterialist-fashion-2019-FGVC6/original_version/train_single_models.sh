#!/bin/sh
#source activate pytorch11_py36
#nohup sh train_single_models.sh > _logs/train_single_models_200527_2.out &
#nohup sh train_single_models.sh poweroff > _logs/train_single_models_200617_1.out &
#set -e
mkdir -p _logs
N_WORKERS=4

#----------------------
# model
#----------------------
#MODEL_TYPE_G=unet4
#MODEL_TYPE_G=unet4_resnet
#MODEL_TYPE_G=unet_fgvc6
MODEL_TYPE_G=deeplab_v3

N_EPOCHES=5
BATCH_SIZE=4
EXPER_NAME=debug
if [ ${EXPER_NAME} = "debug" ] ; then
    N_DISPLAY_STEP=10
    N_DISPLAY_VALID_STEP=50
else
    N_DISPLAY_STEP=100
    N_DISPLAY_VALID_STEP=500
fi
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_test
rm -rf tensorboard/${EXPER_NAME}_valid

python single_models.py \
    --exper_name ${EXPER_NAME} \
    --train_mode train \
    --model_type_G ${MODEL_TYPE_G} \
    --n_diaplay_step ${N_DISPLAY_STEP} --n_display_valid_step ${N_DISPLAY_VALID_STEP} --n_save_epoches 1 \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --image_height 256 --image_width 192 --n_in_channels 3 --n_classes 47 \
    --load_masks_from_dir \
    --n_workers ${N_WORKERS} \
    --debug

#    --data_augument \

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
