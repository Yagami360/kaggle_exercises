#!/bin/sh
#nohup sh single_models.sh > _logs/single_models.out &
#nohup sh single_models.sh poweroff > _logs/single_models.out &
#set -e
mkdir -p _logs

#----------------------
# model
#----------------------
CLASSIFIER=mnist_resnet
N_EPOCHES=100
BATCH_SIZE=64
N_SPLITS=4
EXPER_NAME=single_model_k${N_SPLITS}_da_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --n_splits ${N_SPLITS} --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

#----------------------
# model
#----------------------
CLASSIFIER=mnist_resnet
N_EPOCHES=100
BATCH_SIZE=64
N_SPLITS=4
EXPER_NAME=single_model_k${N_SPLITS}_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --n_splits ${N_SPLITS} --data_augument --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

#----------------------
# model
#----------------------
CLASSIFIER=resnet50
N_EPOCHES=100
BATCH_SIZE=64
N_SPLITS=4
EXPER_NAME=single_model_k${N_SPLITS}_da_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --n_splits ${N_SPLITS} --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

#----------------------
# model
#----------------------
CLASSIFIER=pretrained_resnet50
N_EPOCHES=100
BATCH_SIZE=64
N_SPLITS=4
EXPER_NAME=single_model_k${N_SPLITS}_da_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --n_splits ${N_SPLITS} --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
