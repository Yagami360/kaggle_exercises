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

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# 自作クラス
from dataset import load_dataset, DogsVSCatsDataGen
from networks import ResNet50
from models import EnsembleModelClassifier
from models import ImageClassifierKeras, ImageClassifierSklearn, ImageClassifierXGBoost
from utils import save_checkpoint, load_checkpoint

# XGBoost のデフォルトハイパーパラメーター
params_xgboost = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    "learning_rate" : 0.01,             # ハイパーパラメーターのチューニング時は 0.1 で固定  
    "n_estimators" : 1043,
    'max_depth': 6,                     # 3 ~ 9 : 一様分布に従う。1刻み
    'min_child_weight': 0.47,           # 0.1 ~ 10.0 : 対数が一様分布に従う
    'subsample': 0.8,                   # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'colsample_bytree': 0.8,            # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'gamma': 1.15e-05,                  # 1e-8 ~ 1.0 : 対数が一様分布に従う
    'alpha': 0.0,                       # デフォルト値としておく。余裕があれば変更
    'reg_lambda': 1.0,                  # デフォルト値としておく。余裕があれば変更
    'random_state': 71,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="dog-vs-cat_test", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--submit_message", type=str, default="From Kaggle API Python Script")
    parser.add_argument("--competition_id", type=str, default="dogs-vs-cats-redux-kernels-edition")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--pretrained', action='store_true', help="事前学習済みモデルを行うか否か")
    parser.add_argument('--train_only_fc', action='store_true', help="出力層のみ学習対象にする")
    parser.add_argument('--n_samplings', type=int, default=100000, help="サンプリング数")
    parser.add_argument("--n_splits", type=int, default=2, help="k-fold CV での学習用データセットの分割数")
    parser.add_argument('--batch_size', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=224, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=224, help="入力画像の幅（pixel単位）")
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
    warnings.filterwarnings('ignore')

    # 実行 Device の設定
    if( args.debug ):
        print( "実行デバイス : \n", device_lib.list_local_devices() )

    #======================================================================
    # データセットを読み込みとデータの前処理
    #======================================================================
    X_train, y_train = load_dataset(
        root_dir = args.dataset_dir, datamode =  "train",
        image_height = args.image_height, image_width = args.image_width, n_samplings = args.n_samplings, one_hot_encode = False,
    )

    X_test, _ = load_dataset(
        root_dir = args.dataset_dir, datamode =  "test",
        image_height = args.image_height, image_width = args.image_width, n_samplings = args.n_samplings, one_hot_encode = False,   
    )
    
    y_pred_val = np.zeros((len(y_train),)).astype("int8")

    if( args.debug ):
        print( "X_train.shape : ", X_train.shape )
        print( "y_train.shape : ", y_train.shape )
        print( "X_test.shape : ", X_test.shape )
        print( "y_pred_val.shape : ", y_pred_val.shape )

    # 前処理
    pass

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    pass

    #======================================================================
    # モデルの学習処理
    #======================================================================
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
        y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]
        if( args.debug ):
            print( "X_train_fold.shape={}, X_valid_fold.shape={}".format(X_train_fold.shape, X_valid_fold.shape ) )
            print( "y_train_fold.shape={}, y_valid_fold.shape={}".format(y_train_fold.shape, y_valid_fold.shape ) )

        #--------------------
        # モデル定義
        #--------------------
        # resnet50
        resnet = ResNet50(
            image_height = args.image_height,
            image_width = args.image_width,
            n_channles = 3,
            pretrained = args.pretrained,
            train_only_fc = args.train_only_fc,
        ).finetuned_resnet50
        resnet.compile( loss = 'categorical_crossentropy', optimizer = optimizers.Adam( lr = 0.0001, beta_1 = 0.5, beta_2 = 0.999 ), metrics = ['accuracy'] )
        if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
            load_checkpoint(resnet, args.load_checkpoints_path )

        resnet_classifier = ImageClassifierKeras(
            resnet,
            n_classes = 2,
            debug = args.debug,
        )

        # その他機械学習モデル
        knn_classifier = ImageClassifierSklearn( 
            KNeighborsClassifier( n_neighbors = 2, p = 2, metric = 'minkowski', n_jobs = -1 ),
            debug = args.debug,
        )

        svm_classifier = ImageClassifierSklearn(
            SVC( kernel = 'rbf',gamma = 10.0, C = 0.1, probability = True, random_state = args.seed ),
            debug = args.debug,
        )

        randomforest_classifier = ImageClassifierSklearn(
            RandomForestClassifier( criterion = "gini", bootstrap = True, n_estimators = 1001, oob_score = True, n_jobs = -1, random_state = args.seed ),
            debug = args.debug,
        )

        xgboost_classifier = ImageClassifierXGBoost(
            xgb.XGBClassifier( booster = params_xgboost['booster'],
                objective = params_xgboost['objective'],
                learning_rate = params_xgboost['learning_rate'],
                n_estimators = params_xgboost['n_estimators'],
                max_depth = params_xgboost['max_depth'],
                min_child_weight = params_xgboost['min_child_weight'],
                subsample = params_xgboost['subsample'],
                colsample_bytree = params_xgboost['colsample_bytree'],
                gamma = params_xgboost['gamma'],
                alpha = params_xgboost['alpha'],
                reg_lambda = params_xgboost['reg_lambda'],
                random_state = params_xgboost['random_state']
            ),
            debug = args.debug,
        )

        # アンサンブルモデル
        ensemble_classifier = EnsembleModelClassifier(
            classifiers  = [ resnet_classifier, knn_classifier, svm_classifier, randomforest_classifier, xgboost_classifier ],
            weights = [0.75, 0.25, 0.25, 0.25, 0.50 ],
            fitting = [ False, True, True, True, True ],
            vote_method = "majority_vote",
        )

        #--------------------
        # モデルの学習処理
        #--------------------
        ensemble_classifier.fit( X_train_fold, y_train_fold )

        #--------------------
        # モデルの推論処理
        #--------------------
        if( args.output_type == "proba" ):
            y_pred_test = ensemble_classifier.predict_proba(X_test)[:,1]
        else:
            y_pred_test = ensemble_classifier.predict(X_test)

        y_preds.append(y_pred_test)
        if( args.debug ):
            print( "[{}] y_pred_test[0:10] : {}".format(fold_id, y_pred_test[0:min(10,args.n_samplings)]) )

        y_pred_val[valid_index] = np.argmax( ensemble_classifier.predict_proba(X_valid_fold), axis = 1).astype("int8")

    accuracy = (y_train == y_pred_val).sum()/len(y_pred_val)
    print( "accuracy [val] : {:0.5f}".format(accuracy) )

    y_preds = sum(y_preds) / len(y_preds)   # k-fold CV で平均化 
    y_preds = list(map(float, y_preds))     # 0.999e-01 -> 0.999 の記載にする
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
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
