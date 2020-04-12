import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random
import warnings
import pandas as pd
from matplotlib import pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

# TensorFlow ライブラリ
import tensorflow as tf
from tensorflow.python.framework import ops

# keras
import keras
from keras.models import Sequential
from keras import backend as K
from keras.datasets import mnist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="dog-vs-cat_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="データセットのディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--network_type', choices=['resnet50'], default="resnet50", help="ネットワークの種類")
    parser.add_argument('--pretrained', action='store_true', help="事前学習済みモデルを行うか否か")
    parser.add_argument('--train_only_fc', action='store_true', help="出力層のみ学習対象にする")
    parser.add_argument('--n_steps', type=int, default=10000, help="学習ステップ数")
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=0.0001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")

    parser.add_argument('--image_height', type=int, default=224, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=224, help="入力画像の幅（pixel単位）")
    parser.add_argument('--enable_da', action='store_true', help="Data Augumentation を行うか否か")

    parser.add_argument('--n_display_step', type=int, default=50, help="tensorboard への表示間隔")
    parser.add_argument("--n_save_step", type=int, default=1000, help="モデルのチェックポイントの保存間隔")
    parser.add_argument("--seed", type=int, default=71, help="乱数シード値")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()
    if( args.debug ):
        print( "----------------------------------------------" )
        print( "実行条件" )
        print( "----------------------------------------------" )
        print( "開始時間：", datetime.now() )
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    # 実行 Device の設定
    pass

    # tensorboard
    pass

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)

    #======================================================================
    # データセットを読み込みとデータの前処理
    #======================================================================
    (X_train, y_train),(X_test,y_test) = mnist.load_data()
    
    print( type(X_train) )
    print( "X_train.shape :", X_train.shape )   # (60000, 28, 28)
    print( "y_train.shape :", y_train.shape )   # (60000,)
    print( "X_test.shape :", X_test.shape )     #
    print( "y_test.shape :", y_test.shape )     #