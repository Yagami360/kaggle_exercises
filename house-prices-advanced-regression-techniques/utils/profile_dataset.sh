#!bin/sh
set -e
mkdir -p ../datasets/output

DATASET_FILE=../datasets/input/train.csv
REPORT_FILE=../datasets/output/report_train_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/input/test.csv
REPORT_FILE=../datasets/output/report_test_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/input/sample_submission.csv
REPORT_FILE=../datasets/output/report_sample_submission_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}
