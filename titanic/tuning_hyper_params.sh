#!/bin/sh
set -e

CLASSIFIER=logistic
EXPER_NAME=tuning_${CLASSIFIER}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_trials 100 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

CLASSIFIER=knn
EXPER_NAME=tuning_${CLASSIFIER}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_trials 100 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

CLASSIFIER=svm
EXPER_NAME=tuning_${CLASSIFIER}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_trials 1000 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

CLASSIFIER=random_forest
EXPER_NAME=tuning_${CLASSIFIER}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_trials 1000 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

CLASSIFIER=xgboost
EXPER_NAME=tuning_${CLASSIFIER}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --n_trials 1000 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
