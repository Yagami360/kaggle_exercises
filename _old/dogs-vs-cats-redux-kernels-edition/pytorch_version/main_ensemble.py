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
from dataset import DogsVSCatsDataset, DogsVSCatsDataLoader
from models import ImageClassifierDNN
from models import EnsembleModelClassifier
from utils import save_checkpoint, load_checkpoint


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
    parser.add_argument('--load_checkpoints_path', action='append', help="モデルの読み込みファイルのパス")
    parser.add_argument('--network_type', action='append', choices=['my_resnet18', 'resnet18', 'resnet50'], help="ネットワークの種類")
    parser.add_argument('--image_height', type=int, default=224, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=224, help="入力画像の幅（pixel単位）")
    parser.add_argument('--batch_size', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--n_samplings', type=int, default=100000, help="サンプリング数")
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
    ds_test = DogsVSCatsDataset( args, args.dataset_dir, "test", enable_da = False )
    dloader_test = DogsVSCatsDataLoader(ds_test, batch_size=args.batch_size, shuffle=False, n_workers = args.n_workers )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================    
    resnet_classifier = ImageClassifierDNN( device, args.network_type[0], n_classes = 2, pretrained = False, train_only_fc = True )
    resnet_classifier.load_check_point( args.load_checkpoints_path[0] )

    knn_classifier = KNeighborsClassifier( n_neighbors = 3, p = 2, metric = 'minkowski' )
    svm_classifier = SVC( 
            kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
            gamma = 10.0,       # RFBカーネル関数のγ値
            C = 0.1,            # C-SVM の C 値
            random_state = args.seed,   #
            probability = True  # 学習後の predict_proba method による予想確率を有効にする
    )

    forest = RandomForestClassifier(
                criterion = "gini",     # 不純度関数 [purity]
                bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                n_estimators = 1001,    # 弱識別器（決定木）の数
                n_jobs = -1,            # The number of jobs to run in parallel for both fit and predict ( -1 : 全てのCPUコアで並列計算)
                random_state = args.seed,       #
                oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
            )

    ensemble_classifier = EnsembleModelClassifier(
        classifiers  = [ resnet_classifier, knn_classifier ],
        weights = [0.75, 0.25 ],
        vote_method = "majority_vote",
    )

    #======================================================================
    # モデルの学習 & 推論処理
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
            #save_image( image, os.path.join(args.results_dir, args.exper_name, image_name[0]) )

        # scikit-learn 用にデータ変換
        X_test = image.detach().cpu().numpy()[0]
        y_train = image.detach().cpu().numpy()[0]

        #--------------------
        # モデルの学習処理
        #--------------------
        ensemble_classifier.fit(X_test, y_train)

        #--------------------
        # モデルの推論
        #--------------------
        if( args.output_type == "fixed" ):
            predicts = ensemble_classifier.predict( X_test )
        else:
            predicts = ensemble_classifier.predict_proba( X_test )

        y_preds.append( predicts.tolist()[0] )
        if( args.debug and n_print > 0 ):
            print( "predicts.shape :", predicts.shape )
            print( "predicts[0]={}".format(predicts[0].item()) )
            #print( "y_preds :", y_preds )

        n_print -= 1
        if( step >= args.n_samplings ):
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
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
