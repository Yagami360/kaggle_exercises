# -*- coding:utf-8 -*-
import os
from PIL import Image
import imageio

# keras
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard

#====================================================
# モデルの保存＆読み込み関連
#====================================================
def save_checkpoint(model, save_path):
    model_json = model.to_json()
    open(save_path + ".json", 'w').write(model_json)
    model.save_weights( save_path + ".hdf5" )
    return

def load_checkpoint(model, checkpoint_path):
    model.load_weights( checkpoint_path )
    return

#====================================================
# 画像の保存関連
#====================================================


#====================================================
# TensorBoard への出力関連
#====================================================
