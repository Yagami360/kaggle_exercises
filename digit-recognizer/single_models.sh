#!/bin/sh
#nohup sh single_models.sh > _logs/single_models.out &
#nohup sh single_models.sh poweroff > _logs/single_models.out &
#set -e
mkdir -p _logs

#----------------------
# model1
#----------------------
#<<COMMENTOUT
CLASSIFIER=catboost
EXPER_NAME=single_model_${CLASSIFIER}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --device gpu --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
#COMMENTOUT

#----------------------
# model2
#----------------------
CLASSIFIER=mlp
N_EPOCHES=10
BATCH_SIZE=64
EXPER_NAME=single_model_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --data_augument --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

#----------------------
# model3
#----------------------
CLASSIFIER=resnet50
N_EPOCHES=10
BATCH_SIZE=64
EXPER_NAME=single_model_da_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

#----------------------
# model4
#----------------------
CLASSIFIER=pretrained_resnet50
N_EPOCHES=10
BATCH_SIZE=64
EXPER_NAME=single_model_da_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

#----------------------
# model5
#----------------------
CLASSIFIER=pretrained_resnet50_fc
N_EPOCHES=10
BATCH_SIZE=64
EXPER_NAME=single_model_da_${CLASSIFIER}_ep${N_EPOCHES}_b${BATCH_SIZE}
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_epoches ${N_EPOCHES} --batch_size ${BATCH_SIZE} --debug --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

if [ $1 = "poweroff" ]; then
    sudo poweroff
    sudo shutdown -h now
fi
