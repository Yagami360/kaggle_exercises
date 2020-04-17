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
from tensorflow.python.client import device_lib

# keras
import keras
from keras.preprocessing import image
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend.tensorflow_backend as KTF

# 自作クラス
from dataset import load_dataset, DogsVSCatsDataGen
from networks import ResNet50
from utils import save_checkpoint, load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="dog-vs-cat_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--dataset_dir', type=str, default="../datasets", help="データセットのディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--use_tensorboard', action='store_true', help="TensorBoard 使用")
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
    parser.add_argument('--enable_datagen', action='store_true', help="データジェネレータを使用するか否か")
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
        print( "tensorflow : ", tf.__version__ )
        print( "keras : ", keras.__version__ )
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    # 出力ディレクトリ
    if not( os.path.exists(args.tensorboard_dir) ):
        if( args.use_tensorboard ):
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
    tf.set_random_seed(args.seed)

    # 警告非表示
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', FutureWarning)
    #warnings.filterwarnings('ignore')

    # 実行 Device の設定
    if( args.debug ):
        print( "実行デバイス : \n", device_lib.list_local_devices() )

    # keras で tensorboard を使用するには、keras に session を認識させる必要がある。
    if( args.use_tensorboard ):
        old_session = KTF.get_session()
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)

    #======================================================================
    # データセットを読み込みとデータの前処理
    #======================================================================
    if( args.enable_datagen ):
        datagen_train = DogsVSCatsDataGen( 
            args = args, 
            root_dir = args.dataset_dir, 
            datamode =  "train",
            image_height = args.image_height, image_width = args.image_width, batch_size = args.batch_size,
            debug = args.debug
        )
    else:
        X_train, y_train = load_dataset(
            root_dir = args.dataset_dir, datamode =  "train",
            image_height = args.image_height, image_width = args.image_width,
        )

    # 前処理 & Data Augumentaion
    """
    if( args.enable_datagen ):
        datagen_train = image.ImageDataGenerator(
            rescale = 1.0 / 255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            preprocessing_function = None   # 正規化処理を行なう関数を指定
        )

        datagen_train = datagen_train.flow(
            datagen_train,
            batch_size = args.batch_size,
            seed = args.seed,
        )

        print( datagen_train )
        print( datagen_train.class_indices )
    else:
        pass
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
    # call backs の設定
    #======================================================================
    # 各エポック終了毎のモデルのチェックポイント保存用 call back
    callback_checkpoint = ModelCheckpoint( 
        filepath = os.path.join(args.save_checkpoints_dir, args.exper_name, "step_{epoch:08d}.hdf5"), 
        monitor = 'loss', 
        verbose = 2, 
        save_weights_only = True,   # 
        save_best_only = False,     # 精度がよくなった時だけ保存するかどうか指定。False の場合は毎 epoch 毎保存．
        mode = 'auto',              # 
        period = args.n_save_step   # 何エポックごとに保存するか
    )

    # tensorboard の出力用の call back
    if( args.use_tensorboard ):
        callback_board_train = TensorBoard( log_dir = os.path.join(args.tensorboard_dir, args.exper_name), write_graph  = False )
        callbacks = [ callback_board_train, callback_checkpoint ]
    else:
        callbacks = [ callback_checkpoint ]

    #======================================================================
    # モデルの学習処理
    #======================================================================
    if( args.enable_datagen ):
        history = model.fit_generator( 
            generator = datagen_train, 
            epochs = args.n_steps, 
            steps_per_epoch = len(datagen_train),
            verbose = 1,
            workers = args.n_workers,
            shuffle = True,
            use_multiprocessing = True,
            callbacks = callbacks
        )
    else:
        history = model.fit( 
            x = X_train, y = y_train, 
            epochs = args.n_steps, 
            steps_per_epoch = 1,
            verbose = 1,
            workers = args.n_workers,
            shuffle = True,
            use_multiprocessing = True,
            callbacks = callbacks

    # モデルの保存
    save_checkpoint( model, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final') )

    #======================================================================
    # モデルの評価
    #======================================================================
    print( history.history.keys() )
    print( history.history['accuracy'][0:10] )

    # session を元に戻す（tensorboard用）
    if( args.use_tensorboard ):
        KTF.set_session(old_session)
