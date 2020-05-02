#!bin/sh
set -e
mkdir -p ../datasets

DATASET_FILE=../datasets/bureau.csv
REPORT_FILE=../datasets/report_bureau_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/credit_card_balance.csv
REPORT_FILE=../datasets/report_credit_card_balance_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/POS_CASH_balance.csv
REPORT_FILE=../datasets/report_POS_CASH_balance_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/application_test.csv
REPORT_FILE=../datasets/report_application_test_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}

DATASET_FILE=../datasets/application_train.csv
REPORT_FILE=../datasets/report_application_train_csv.html
pandas_profiling ${DATASET_FILE} ${REPORT_FILE}
