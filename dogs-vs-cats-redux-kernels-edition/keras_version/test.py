import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random
import warnings
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
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
    parser.add_argument("--exper_name", default="dog-vs-cat_test", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="dogs-vs-cats-redux-kernels-edition")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--network_type', choices=['resnet50'], default="resnet50", help="ネットワークの種類")
    parser.add_argument('--pretrained', action='store_true', help="事前学習済みモデルを行うか否か")
    parser.add_argument('--train_only_fc', action='store_true', help="出力層のみ学習対象にする")
    parser.add_argument('--n_samplings', type=int, default=100000, help="サンプリング数")
    parser.add_argument('--batch_size', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=224, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=224, help="入力画像の幅（pixel単位）")
    parser.add_argument('--enable_datagen', action='store_true', help="データジェネレータを使用するか否か")
    parser.add_argument('--enable_da', action='store_true', help="Data Augumentation を行うか否か")
    parser.add_argument('--output_type', choices=['fixed', 'proba'], default="proba", help="主力形式（確定値 or 確率値）")
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
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))

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

        datagen_test = DogsVSCatsDataGen( 
            args = args, 
            root_dir = args.dataset_dir, 
            datamode =  "test",
            image_height = args.image_height, image_width = args.image_width, batch_size = args.batch_size,
            debug = args.debug
        )

    else:
        X_train, y_train = load_dataset(
            root_dir = args.dataset_dir, datamode =  "train",
            image_height = args.image_height, image_width = args.image_width, n_samplings = args.n_samplings,
        )

        X_test, _ = load_dataset(
            root_dir = args.dataset_dir, datamode =  "test",
            image_height = args.image_height, image_width = args.image_width, n_samplings = args.n_samplings,    
        )
        
        if( args.debug ):
            print( "X_train.shape : ", X_train.shape )
            print( "y_train.shape : ", y_train.shape )
            print( "X_test.shape : ", X_test.shape )

    # 前処理
    pass

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
    # optimizer, loss を設定（ダミー）
    #======================================================================
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizers.Adam( lr = 0.0001, beta_1 = 0.5, beta_2 = 0.999 ),
        metrics = ['accuracy']
    )
    if( args.debug ):
        model.summary()

    #======================================================================
    # モデルの推論処理
    #======================================================================
    """
    # 学習用データセットでの accuracy
    if( args.enable_datagen ):
        evals = model.evaluate_generator( datagen_train, steps = min(len(datagen_train.image_names),args.n_samplings), workers = args.n_workers, use_multiprocessing = True, verbose = 1 )
    else:
        evals = model.evaluate( x = X_train, y = y_train, batch_size = args.batch_size, workers = args.n_workers, use_multiprocessing = True, verbose = 1 )

    print( "loss [train] : ", evals[0] )
    print( "accuracy [train] : ", evals[1] )
    """

    # 予想ラベルを推論
    if( args.enable_datagen ):
        predicts = model.predict_generator( datagen_test, steps = min(len(datagen_train.image_names),args.n_samplings), workers = args.n_workers, use_multiprocessing = True, verbose = 1 )    
    else:
        predicts = model.predict( X_test, batch_size = args.batch_size, workers = args.n_workers, use_multiprocessing = True, verbose = 1 ) 

    if( args.output_type == "fixed" ):
        y_preds = np.argmax(predicts, axis = 1)
    else:
        y_preds = predicts[:,1]          

    y_preds = list(map(float, y_preds)) # 0.999e-01 -> 0.999 の記載にする
    print( "len(y_preds)", len(y_preds) )
    print( "y_preds[0:10]", y_preds[0:min(10,args.n_samplings)] )

    #================================
    # 可視化処理
    #================================
    classes = {0: 'cat', 1: 'dog'}
    fig, axes = plt.subplots(5, 5, figsize=(16, 20), facecolor='w')
    for i, ax in enumerate(axes.ravel()):
        if y_preds[i] > 0.5:
            label = 1
        else:
            label = 0
            
        ax.set_title( '{}.jpg'.format(i+1) + " / " + classes[label])
        img = Image.open( os.path.join(args.dataset_dir, "test", '{}.jpg'.format(i+1)) )
        ax.imshow(img)

    fig.savefig( os.path.join(args.results_dir, args.exper_name, "classification.png"), dpi = 150, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    ds_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )
    ds_submission['label'][0:len(y_preds)] = y_preds
    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
