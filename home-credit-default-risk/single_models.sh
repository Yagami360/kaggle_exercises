#!/bin/sh
set -e

#------------------------
# Exper
#------------------------
<<COMMENTOUT
CLASSIFIER=catboost
EXPER_NAME=single_model_${CLASSIFIER}_iter1000_invalidFeat_timeFeat
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --polynomial_features --domain_features --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
COMMENTOUT

#------------------------
# Exper
#------------------------
<<COMMENTOUT
CLASSIFIER=catboost
EXPER_NAME=single_model_${CLASSIFIER}_iter1000_invalidFeat_timeFeat_domainFeat
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --polynomial_features --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
COMMENTOUT

#------------------------
# Exper
#------------------------
<<COMMENTOUT
CLASSIFIER=catboost
EXPER_NAME=single_model_${CLASSIFIER}_iter1000_invalidFeat_timeFeat_polynomialFeat_domainFeat
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
COMMENTOUT

#------------------------
# Exper
#------------------------
CLASSIFIER=catboost
EXPER_NAME=single_model_${CLASSIFIER}_iter1000_onehot_invalidFeat_timeFeat_domainFeat
mkdir -p results/${EXPER_NAME}
python single_models.py --exper_name ${EXPER_NAME} --classifier ${CLASSIFIER} --onehot --polynomial_features --submit > results/${EXPER_NAME}/logs_${EXPER_NAME}.out
