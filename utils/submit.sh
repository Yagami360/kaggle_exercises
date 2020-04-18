#!bin/sh
set -e
COMPETITIONS_NAME=titanic
EXPER_NAME=debug
FILE_NAME=../${COMPETITIONS_NAME}/results/${EXPER_NAME}/submission.csv
MESSAGE="From Kaggle API"

kaggle competitions submissions -c ${COMPETITIONS_NAME}
kaggle competitions submit -c ${COMPETITIONS_NAME} -f ${FILE_NAME} -m ${MESSAGE}
kaggle competitions submissions -c ${COMPETITIONS_NAME}
