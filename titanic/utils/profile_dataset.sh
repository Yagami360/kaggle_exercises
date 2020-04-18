#!bin/sh
set -e
mkdir -p ../datasets

DATASET_FILE=../datasets/train.csv
REPORT_FILE=../datasets/report_train_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/test.csv
REPORT_FILE=../datasets/report_test_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/gender_submission.csv
REPORT_FILE=../datasets/report_gender_submission_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}
