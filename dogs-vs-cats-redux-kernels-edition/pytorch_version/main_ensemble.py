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
from kaggle.api.kaggle_api_extended import KaggleApi
#from apex import amp

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# NN 以外の機械学習モデル
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# 自作クラス
from dataset import DogsVSCatsDataset, DogsVSCatsDataLoader, load_dataset
from networks import MyResNet18, ResNet18, ResNet50
from models import ImageClassifierPyTorch, ImageClassifierSklearn, ImageClassifierXGBoost
from models import EnsembleModelClassifier
from utils import save_checkpoint, load_checkpoint

# XGBoost のデフォルトハイパーパラメーター
params_xgboost = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    "learning_rate" : 0.01,             # ハイパーパラメーターのチューニング時は 0.1 で固定  
    "n_estimators" : 100,
    'max_depth': 3,                     # 3 ~ 9 : 一様分布に従う。1刻み
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
    parser.add_argument("--exper_name", default="emsemble_resnet_sklearn_xgboost", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="dogs-vs-cats-redux-kernels-edition")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--load_checkpoints_path', action='append', help="モデルの読み込みファイルのパス")
    parser.add_argument('--network_type', action='append', choices=['my_resnet18', 'resnet18', 'resnet50'], help="ネットワークの種類")
    parser.add_argument('--image_height', type=int, default=224, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=224, help="入力画像の幅（pixel単位）")
    parser.add_argument('--batch_size', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--n_train_samplings', type=int, default=100000, help="学習用データのサンプリング数")
    parser.add_argument('--n_test_samplings', type=int, default=100000, help="テスト用データのサンプリング数")
    parser.add_argument('--n_fmaps', type=int, default=64, help="１層目の特徴マップの枚数")
    parser.add_argument('--enable_da', action='store_true', help="Data Augumentation を行うか否か")
    parser.add_argument('--output_type', choices=['fixed', 'prob'], default="prob", help="主力形式（確定値 or 確率値）")

    parser.add_argument('--use_amp', action='store_true', help="AMP [Automatic Mixed Precision] の使用有効化")
    parser.add_argument('--opt_level', choices=['O0','O1','O2','O3'], default='O1', help='mixed precision calculation mode')
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))

    if( args.debug ):
        print( "----------------------------------------------" )
        print( "実行条件" )
        print( "----------------------------------------------" )
        print( "開始時間：", datetime.now() )
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    # 実行 Device の設定
    if( args.device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            #torch.cuda.set_device(args.gpu_ids[0])
            print( "実行デバイス :", device)
            print( "GPU名 :", torch.cuda.get_device_name(device))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device)
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device)

    # seed 値の固定
    if( args.use_cuda_deterministic ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    X_train, y_train = load_dataset(
        root_dir = args.dataset_dir, datamode =  "train",
        image_height = args.image_height, image_width = args.image_width, n_samplings = args.n_train_samplings,
    )

    """
    X_test, _ = load_dataset(
        root_dir = args.dataset_dir, datamode =  "test",
        image_height = args.image_height, image_width = args.image_width, n_samplings = args.n_test_samplings,   
    )
    """

    ds_test = DogsVSCatsDataset( args, args.dataset_dir, "test", enable_da = False )
    dloader_test = DogsVSCatsDataLoader(ds_test, batch_size=args.batch_size, shuffle=False, n_workers = args.n_workers )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================    
    if( args.network_type[0] == "my_resnet18" ):
        resnet = MyResNet18( n_in_channels = 3, n_fmaps = 64, n_classes = 2 ).to(device)
    elif( args.network_type[0] == "resnet18" ):
        resnet = ResNet18( n_classes = 2, pretrained = False, train_only_fc = False ).to(device)
    else:
        resnet = ResNet50( n_classes = 2, pretrained = False, train_only_fc = False ).to(device)

    resnet_classifier = ImageClassifierPyTorch( device, resnet, debug = args.debug )
    resnet_classifier.load_check_point( args.load_checkpoints_path[0] )

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
        weights = [0.70, 0.05, 0.05, 0.10, 0.10 ],
        fitting = [ False, True, True, True, True ],
        vote_method = "majority_vote",
    )

    #======================================================================
    # モデルの学習処理
    #======================================================================
    ensemble_classifier.fit(X_train, y_train)

    #======================================================================
    # モデルの推論処理
    #======================================================================
    y_preds = []
    n_print = 5
    for step, inputs in enumerate(tqdm(dloader_test.data_loader)):
        # ミニバッチデータを GPU へ転送
        inputs = dloader_test.next_batch()
        image_name = inputs["image_name"]
        image = inputs["image"].to(device)
        targets = inputs["targets"].to(device)
        if( args.debug and n_print > 0):
            print( "image_name : ", image_name )
            print( "image.shape : ", image.shape )
            print( "targets.shape : ", targets.shape )

        #--------------------
        # モデルの推論
        #--------------------
        if( args.output_type == "fixed" ):
            predicts = ensemble_classifier.predict( image )
        else:
            predicts = ensemble_classifier.predict_proba( image )[:,1]

        y_preds.append( predicts.tolist()[0] )
        if( args.debug and n_print > 0 ):
            print( "predicts.shape :", predicts.shape )
            print( "predicts[0]={}".format(predicts[0]) )
            #print( "y_preds :", y_preds )

        n_print -= 1
        if( step >= args.n_test_samplings ):
            break

    print( "y_preds[0:10] :\n", y_preds[0:10] )
    print( "len(y_preds) : ", len(y_preds) )

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

    fig.savefig( os.path.join(args.results_dir, args.exper_name, "classification.png"), dpi = 100, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    ds_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )
    ds_submission['label'][0:len(y_preds)] = list(map(float, y_preds))
    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.dataset_dir, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
