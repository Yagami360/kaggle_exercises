#!/bin/sh
#source activate tensorflow_p36
mkdir -p tensorboard
nohup tensorboard --logdir tensorboard --port 6006 &
