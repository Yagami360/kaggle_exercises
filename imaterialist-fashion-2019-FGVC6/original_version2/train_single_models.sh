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
MODEL_TYPE_G=unet_fgvc6

N_EPOCHES=5
BATCH_SIZE=4
EXPER_NAME=debug
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_test
rm -rf tensorboard/${EXPER_NAME}_valid

python single_models.py \
    --train_mode train \
    --model_type_G ${MODEL_TYPE_G} \
    --n_diaplay_step 100 --n_display_valid_step 500 --n_save_epoches 1 \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --image_height 256 --image_width 192 --n_channels 3 --n_classes 47 \
    --lambda_l1 0.0 --lambda_vgg 0.0 --lambda_entropy 1.0 --lambda_parsing_entropy 0.0 \
    --load_masks_from_dir \
    --n_workers ${N_WORKERS} \
    --debug

#    --exper_name ${EXPER_NAME} \
#    --data_augument \

if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
