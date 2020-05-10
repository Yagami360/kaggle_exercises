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
def convert_rle(img, order='F', format=True):
    """
    画像を連長圧縮 [RLE : Run Length Encoding] にエンコードする。
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not
    returns run length as an array or string (if format is True)
    Source https://www.kaggle.com/bguberfain/unet-with-depth
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

#====================================================
# TensorBoard への出力関連
#====================================================
