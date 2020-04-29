#!/bin/sh
set -e

<<COMMENTOUT
REGRESSOR=logistic
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 20 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/

REGRESSOR=knn
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 50 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/

REGRESSOR=svm
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 500 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/

REGRESSOR=random_forest
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 100 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/

REGRESSOR=bagging
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 100 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/

REGRESSOR=adaboost
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 100 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/

REGRESSOR=xgboost
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 500 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/
COMMENTOUT

REGRESSOR=lightgbm
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 500 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/

REGRESSOR=catboost
EXPER_NAME=tuning_${REGRESSOR}
mkdir -p results/${EXPER_NAME}
python tuning_hyper_params.py --exper_name ${EXPER_NAME} --regressor ${REGRESSOR} --n_trials 500 > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
cp -r results/${EXPER_NAME}/${EXPER_NAME}.yml parames/