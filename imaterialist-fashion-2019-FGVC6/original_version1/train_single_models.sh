#!/bin/sh
#source activate pytorch11_py36
#nohup sh train_single_models.sh > _logs/train_single_models_200527.out &
#nohup sh train_single_models.sh > _logs/train_single_models_200527_1.out &
#nohup sh train_single_models.sh > _logs/train_single_models_200527_2.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
#MODEL_TYPE_G=unet4
#MODEL_TYPE_G=unet4resnet34
MODEL_TYPE_G=unet_fgvc6
#MODEL_TYPE_G=mgvton
MODEL_TYPE_D=patchgan

N_EPOCHES=100
BATCH_SIZE=4
#EXPER_NAME=debug
#rm -rf tensorboard/${EXPER_NAME}

python single_models.py \
    --train_mode train \
    --model_type_G ${MODEL_TYPE_G} --model_type_D ${MODEL_TYPE_D} \
    --n_diaplay_step 100 --n_display_valid_step 100 --n_save_epoches 10 \
    --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} \
    --image_height 256 --image_width 192 --n_channels 3 \
    --lambda_l1 0.0 --lambda_vgg 0.0 --lambda_bce 1.0 --lambda_enpropy 0.0 --lambda_adv 0.0 \
    --debug

#    --lambda_l1 5.0 --lambda_vgg 5.0 --lambda_bce 0.0 --lambda_enpropy 0.0 --lambda_adv 1.0 \
#    --exper_name ${EXPER_NAME} \
#    --data_augument \


if [ $1 = "poweroff" ] ; then
    sudo poweroff
    sudo shutdown -h now
fi
