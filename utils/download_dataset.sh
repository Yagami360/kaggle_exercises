#!bin/sh
set -e
COMPETITIONS_NAME=titanic

mkdir -p ../${COMPETITIONS_NAME}
mkdir -p ../${COMPETITIONS_NAME}/input

cd ../${COMPETITIONS_NAME}/input
kaggle competitions files -c ${COMPETITIONS_NAME}
kaggle competitions download -c ${COMPETITIONS_NAME}
unzip ${COMPETITIONS_NAME}.zip
rm -rf ${COMPETITIONS_NAME}.zip
