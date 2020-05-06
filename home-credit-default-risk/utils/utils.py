# -*- coding:utf-8 -*-
import os
from PIL import Image
import imageio
import pandas as pd
import feather

# keras
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard

#====================================================
# データセット関連
#====================================================
def read_feature( file_path ):
    if not( os.path.exists( file_path ) ):
        df_data = pd.read_csv( file_path.split(".feature")[0] + ".csv" )
        #df_data.to_feather( file_path )
        feather.write_dataframe(df_data, file_path)
    else:
        #df_data = pd.read_feather( file_path )
        df_data = feather.read_dataframe(file_path)

    return df_data

def save_feature( df_data, file_path ):
    feather.write_dataframe(df_data, file_path)
    return

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
