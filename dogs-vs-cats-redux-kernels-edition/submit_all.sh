#!/bin/sh
#source activate pytorch11_py36
#nohup sh submit_all.sh > _logs/submit_all.out &
set -e
ROOT_DIR=${PWD}

#-----------------------------
# submit1
#-----------------------------
<<COMMENTOUT
cd ${ROOT_DIR}/pytorch_version
NETWORK_TYPE=my_resnet18
LOAD_CHECKPOINTS_PATH=checkpoints/my_resnet18_b64_norm_da_200411/model_final.pth
EXEP_NAME=${NETWORK_TYPE}_b64_norm_da_pytorch
python test.py \
    --exper_name ${EXEP_NAME} \
    --submit_message ${EXEP_NAME} \
    --device gpu \
    --dataset_dir datasets \
    --network_type ${NETWORK_TYPE} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --batch_size 1 \
    --n_samplings 100000 \
    --debug \
    --submit
COMMENTOUT
cd ..

#-----------------------------
# submit2
#-----------------------------
<<COMMENTOUT
cd ${ROOT_DIR}/pytorch_version
NETWORK_TYPE=resnet50
LOAD_CHECKPOINTS_PATH=checkpoints/${NETWORK_TYPE}_b32_norm_da_200411/model_final.pth
EXEP_NAME=${NETWORK_TYPE}_b32_norm_da_pytorch
python test.py \
    --exper_name ${EXEP_NAME} \
    --submit_message ${EXEP_NAME} \
    --device gpu \
    --dataset_dir datasets \
    --network_type ${NETWORK_TYPE} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --batch_size 1 \
    --n_samplings 100000 \
    --debug \
    --submit
COMMENTOUT

#-----------------------------
# submit3
#-----------------------------
<<COMMENTOUT
cd ${ROOT_DIR}/pytorch_version
NETWORK_TYPE=resnet50
LOAD_CHECKPOINTS_PATH=checkpoints/${NETWORK_TYPE}_fc_b32_norm_da_200411/model_final.pth
EXEP_NAME=${NETWORK_TYPE}_fc_b32_norm_da_pytorch
python test.py \
    --exper_name ${EXEP_NAME} \
    --submit_message ${EXEP_NAME} \
    --device gpu \
    --dataset_dir datasets \
    --network_type ${NETWORK_TYPE} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --batch_size 1 \
    --n_samplings 100000 \
    --debug \
    --submit
COMMENTOUT

#-----------------------------
# submit
#-----------------------------
cd ${ROOT_DIR}/pytorch_version
NETWORK_TYPE1=my_resnet18
NETWORK_TYPE2=resnet50
NETWORK_TYPE3=resnet50
LOAD_CHECKPOINTS_PATH1=checkpoints/my_resnet18_b64_norm_da_200411/model_final.pth
LOAD_CHECKPOINTS_PATH2=checkpoints/resnet50_b32_norm_da_200411/model_final.pth
LOAD_CHECKPOINTS_PATH3=checkpoints/resnet50_fc_b32_norm_da_200411/model_final.pth
EXEP_NAME=ensemble_resnet_pytorch

python test_ensemble.py \
    --exper_name ${EXEP_NAME} \
    --device gpu \
    --dataset_dir ../datasets \
    --network_type ${NETWORK_TYPE1} --network_type ${NETWORK_TYPE2} --network_type ${NETWORK_TYPE3} \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH1} --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH2} --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH3} \
    --batch_size 1 \
    --n_samplings 100000 \
    --debug \
    --submit

#-----------------------------
# submit
#-----------------------------
cd ${ROOT_DIR}/keras_version
NETWORK_TYPE=resnet50
LOAD_CHECKPOINTS_PATH=checkpoints/${NETWORK_TYPE}_fc_b32_200415/model_final.hdf5
EXEP_NAME=${NETWORK_TYPE}_fc_b32_keras

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

#-----------------------------
# submit
#-----------------------------
cd ${ROOT_DIR}/keras_version

NETWORK_TYPE=resnet50
LOAD_CHECKPOINTS_PATH=checkpoints/resnet50_fc_b32_200415/model_final.hdf5
EXEP_NAME=ensemble_resnet_and_sklearn_xgbbost_keras
python main_ensemble.py \
    --exper_name ${EXEP_NAME} \
    --submit_message ${EXEP_NAME} \
    --device gpu \
    --dataset_dir ../datasets \
    --load_checkpoints_path ${LOAD_CHECKPOINTS_PATH} \
    --train_only_fc \
    --batch_size 1 \
    --n_samplings 100000 \
    --n_splits 2 \
    --debug \
    --submit


sleep 300

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi