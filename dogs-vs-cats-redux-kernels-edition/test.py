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

# 自作クラス
from dataset import DogsVSCatsDataset, DogsVSCatsDataLoader
from networks import ResNet18
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
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--image_height', type=int, default=224, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=224, help="入力画像の幅（pixel単位）")
    parser.add_argument('--batch_size', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--n_samplings', type=int, default=100000, help="サンプリング数")
    parser.add_argument('--n_fmaps', type=int, default=64, help="１層目の特徴マップの枚数")
    parser.add_argument('--enable_da', action='store_true', help="Data Augumentation を行うか否か")

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
    dloader_test = DogsVSCatsDataLoader(ds_test, batch_size=args.batch_size, shuffle=False )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model = ResNet18(
            n_in_channels = 3,
            n_fmaps = args.n_fmaps,
            n_classes = 2
    ).to(device)

    if( args.debug ):
        print( "model :\n", model )

    # モデルを読み込む
    if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
        load_checkpoint(model, device, args.load_checkpoints_path )
        print( "load check points" )

    #======================================================================
    # モデルの推論処理
    #======================================================================
    print("Starting Test Loop...")
    y_preds = []
    n_print = 10
    for step, inputs in enumerate(tqdm(dloader_test.data_loader)):
        model.eval()

        # ミニバッチデータを GPU へ転送
        inputs = dloader_test.next_batch()
        image_name = inputs["image_name"]
        image = inputs["image"].to(device)
        targets = inputs["targets"].to(device)
        if( args.debug and n_print > 0):
            print( "image.shape : ", image.shape )
            print( "targets.shape : ", targets.shape )

        #----------------------------------------------------
        # データをモデルに流し込む
        #----------------------------------------------------
        with torch.no_grad():
            output = model( image )
            if( args.debug and n_print > 0 ):
                print( "output.shape :", output.shape )

        #----------------------------------------------------
        # 正解率を計算する。（バッチデータ）
        #----------------------------------------------------
        # 確率値が最大のラベル 0~9 を予想ラベルとする。
        # dim = 1 ⇒ 列方向で最大値をとる
        # Returns : (Tensor, LongTensor)
        _, predicts = torch.max( output.data, dim = 1 )
        y_preds = np.hstack( (predicts.detach().cpu().numpy(), y_preds) )
        if( args.debug and n_print > 0 ):
            print( "predicts.shape :", predicts.shape )
            print( "output[0]=({:.5f},{:.5f}), predicts[0]={}".format(output[0,0].item(), output[0,1].item(), predicts[0].item()) )

        n_print -= 1
        if( step >= args.n_samplings ):
            break

    print( "y_preds :\n", y_preds )
    print( "y_preds.shape : ", y_preds.shape )

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
