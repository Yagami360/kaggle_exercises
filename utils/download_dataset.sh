#!bin/sh
set -e
#COMPETITIONS_NAME=titanic
#COMPETITIONS_NAME=house-prices-advanced-regression-techniques
COMPETITIONS_NAME=dogs-vs-cats-redux-kernels-edition

mkdir -p ../${COMPETITIONS_NAME}
mkdir -p ../${COMPETITIONS_NAME}/datasets
mkdir -p ../${COMPETITIONS_NAME}/datasets/input

cd ../${COMPETITIONS_NAME}/datasets/input
kaggle competitions files -c ${COMPETITIONS_NAME}
kaggle competitions download -c ${COMPETITIONS_NAME}
unzip ${COMPETITIONS_NAME}.zip
rm -rf ${COMPETITIONS_NAME}.zip
