IMAGE_DIR=../../datasets/test
RESULT_DIR=${PWD}/results
rm -rf ${RESULT_DIR}
mkdir -p ${RESULT_DIR}

cd graphonomy_wrapper
python inference_all.py \
    --device cpu \
    --in_image_dir ${IMAGE_DIR} \
    --results_dir ${RESULT_DIR} \
    --load_checkpoints_path checkpoints/universal_trained.pth \
    --save_vis
