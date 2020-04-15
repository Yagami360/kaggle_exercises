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
from keras.callbacks import ModelCheckpoint, TensorBoard

# 自作クラス
from dataset import DogsVSCatsDataset 
from networks import ResNet50
from utils import save_checkpoint, load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="dog-vs-cat_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--dataset_dir', type=str, default="../datasets", help="データセットのディレクトリ")
    #parser.add_argument('--dataset_dir', type=str, default="/Users/sakai/GitHub/kaggle_exercises/dogs-vs-cats-redux-kernels-edition/datasets", help="データセットのディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    #parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--load_checkpoints_path', type=str, default="checkpoints/dog-vs-cat_train/epoch_00000.h5", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--network_type', choices=['resnet50'], default="resnet50", help="ネットワークの種類")
    parser.add_argument('--pretrained', action='store_true', help="事前学習済みモデルを行うか否か")
    parser.add_argument('--train_only_fc', action='store_true', help="出力層のみ学習対象にする")
    parser.add_argument('--n_epochs', type=int, default=2, help="学習ステップ数")
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

    # 出力ディレクトリ
    if not( os.path.exists(args.tensorboard_dir) ):
        os.mkdir(args.tensorboard_dir)
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )
    if ( os.path.exists("_debug") == False and args.debug ):
        os.mkdir( "_debug" )

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 警告非表示
    warnings.simplefilter('ignore', DeprecationWarning)

    # 実行 Device の設定
    pass

    # tensorboard の出力用の call back
    callback_board_train = TensorBoard( log_dir = os.path.join(args.tensorboard_dir, args.exper_name), histogram_freq = 1 )
    #callback_board_test = TensorBoard( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "test"), histogram_freq = 1 )

    # 各エポック終了毎のモデルのチェックポイント保存用 call back
    callback_checkpoint = ModelCheckpoint( 
        filepath = os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name, "epoch_{epoch:04d}.hd5")), 
        monitor = 'val_loss', 
        verbose = 1, 
        save_best_only = False,     # 精度がよくなった時だけ保存するかどうか指定。Falseの場合は毎epoch保存．
        mode = 'auto'
    )

    #======================================================================
    # データセットを読み込みとデータの前処理
    #======================================================================
    ds_train = DogsVSCatsDataset( 
        args = args, 
        root_dir = args.dataset_dir, 
        datamode =  "train",
        image_height = args.image_height, image_width = args.image_width, batch_size = args.batch_size,
        debug = args.debug
    )

    # 前処理 & Data Augumentaion
    """
    datagen_train = image.ImageDataGenerator(
        rescale = 1.0 / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        preprocessing_function = None   # 正規化処理を行なう関数を指定
    )

    ds_train = datagen_train.flow(
        ds_train,
        batch_size = args.batch_size,
        seed = args.seed,
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

    # モデルを読み込む
    if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
        load_checkpoint(model, args.load_checkpoints_path )

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
        verbose = 2,                        # 進行状況メッセージ出力モード
        workers = args.n_workers,
        shuffle = True,
        use_multiprocessing = True,
        callbacks = [callback_checkpoint, callback_board_train]
    )

    # モデルの保存
    save_checkpoint( model, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final') )

    #======================================================================
    # モデルの評価
    #======================================================================
    """
    evals = model.evaluate( 
                x = X_test, y = y_test, 
                verbose = 1                 # 進行状況メッセージ出力モードで，0か1．
            )
    """