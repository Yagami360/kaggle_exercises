#!/bin/sh
set -e

CLASSIFIER=logistic
PARAMS_FILE=logstic_classifier_titanic.yml
EXPER_NAME=single_model_logstic_tuned
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --params_file parames/${PARAMS_FILE} > results/${EXPER_NAME}/logs_${EXPER_NAME}.out

