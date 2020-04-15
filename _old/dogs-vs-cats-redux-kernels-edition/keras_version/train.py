import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random
import warnings
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from kaggle.api.kaggle_api_extended import KaggleApi

# TensorFlow ライブラリ
import tensorflow as tf

# keras
import keras
from keras.preprocessing import image
from keras import optimizers

# 自作クラス
from dataset import DogsVSCatsDataset 
from networks import ResNet50

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="dog-vs-cat_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--dataset_dir', type=str, default="../datasets", help="データセットのディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--network_type', choices=['resnet50'], default="resnet50", help="ネットワークの種類")
    parser.add_argument('--pretrained', action='store_true', help="事前学習済みモデルを行うか否か")
    parser.add_argument('--train_only_fc', action='store_true', help="出力層のみ学習対象にする")
    parser.add_argument('--n_epochs', type=int, default=10, help="学習ステップ数")
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

    # 警告非表示
    warnings.simplefilter('ignore', DeprecationWarning)
    
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
    ds_train = DogsVSCatsDataset( args, args.dataset_dir, "train", debug = args.debug )
    ds_test = DogsVSCatsDataset( args, args.dataset_dir, "test", debug = args.debug )

    """
    train_image_path = os.path.join( args.dataset_dir, "train" )
    train_image_names = sorted( [f for f in os.listdir(train_image_path) if f.endswith(".jpg")], key=lambda s: int(re.search(r'\d+', s).group()) )
    print( "train_image_names[0:5] : ", train_image_names[0:5] )

    # X_train
    X_train = np.zeros( (len(train_image_names), 3, args.image_height, args.image_width), dtype=np.uint8 )
    for i, name in enumerate(train_image_names):
        img = cv2.imread( os.path.join(train_image_path,name) )
        img = cv2.resize( img, (args.image_height, args.image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        X_train[i] = img.transpose(2,0,1)

    print( "X_train.shape : ", X_train.shape )

    # y_train
    y_train = np.zeros( (len(train_image_names), 1), dtype=np.uint8 )
    for i, name in enumerate(train_image_names):
        if( "cat." in name ):
            y_train[i] = 0
        else:
            y_train[i] = 1

    print( "y_train.shape : ", y_train.shape )
    print( "y_train[0:5] : ", y_train[0:5] )
    """

    """
    # ImageDataGenerator を使用して Data Augumentaion
    datagen_train = image.ImageDataGenerator(
        rescale = 1.0 / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True   
    )

    ds_train = datagen_train.flow_from_directory(
        os.path.join( args.dataset_dir, "train" ),
        target_size = (args.image_height, args.image_width ),
        batch_size = args.batch_size,
        class_mode = 'binary'
    )

    print( ds_train )
    print( ds_train.class_indices )
    """

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model = ResNet50(
        image_height = args.image_height,
        image_width = args.image_width,
        n_channles = 3,
        pretrained = args.pretrained,
        train_only_fc = args.train_only_fc,
    ).finetuned_resnet50

    #======================================================================
    # optimizer, loss を設定
    #======================================================================
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizers.Adam( lr = args.lr, beta_1 = args.beta1, beta_2 = args.beta2 ),
        metrics = ['accuracy']
    )

    if( args.debug ):
        model.summary()

    #======================================================================
    # モデルの学習処理
    #======================================================================
    model.fit_generator( 
        generator = ds_train, 
        epochs = args.n_epochs, 
        steps_per_epoch = len(ds_train),
        verbose = 1, 
        workers = args.n_workers,
        shuffle = True
    )

    #======================================================================
    # モデルの評価
    #======================================================================
    """
    evals = model.evaluate( 
                x = X_test, y = y_test, 
                verbose = 1                 # 進行状況メッセージ出力モードで，0か1．
            )
    """