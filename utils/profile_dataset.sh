#!bin/sh
set -e
COMPETITIONS_NAME=house-prices-advanced-regression-techniques
DATASET_FILE=../${COMPETITIONS_NAME}/datasets/train.csv
REPORT_FILE=../${COMPETITIONS_NAME}/datasets/report_train_csv.html

mkdir -p ../${COMPETITIONS_NAME}
mkdir -p ../${COMPETITIONS_NAME}/datasets

pandas_profiling ${DATASET_FILE} ${REPORT_FILE}