import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random
import warnings
from matplotlib import pyplot as plt
import pandas as pd
#from apex import amp
from kaggle.api.kaggle_api_extended import KaggleApi

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision      # 画像処理関連
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="my-vton_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="データセットのディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--load_checkpoints_tom_path', type=str, default="", help="TOMモデルの読み込みファイルのパス")
    parser.add_argument('--load_checkpoints_D_path', type=str, default="", help="識別器モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--n_steps', type=int, default=100000, help="学習ステップ数")
    parser.add_argument('--batch_size', type=int, default=4, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=4, help="test データのバッチサイズ")
    parser.add_argument('--lr', type=float, default=0.0001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_height', type=int, default=256, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=192, help="入力画像の幅（pixel単位）")

    parser.add_argument('--n_display_step', type=int, default=500, help="tensorboard への表示間隔")
    parser.add_argument('--n_display_test_step', type=int, default=5000, help="test データの tensorboard への表示間隔")
    parser.add_argument('--n_display_val_step', type=int, default=500, help="val データの tensorboard への表示間隔")
    parser.add_argument("--n_save_step", type=int, default=25000, help="モデルのチェックポイントの保存間隔")
    parser.add_argument("--seed", type=int, default=8, help="乱数シード値")
    parser.add_argument('--use_amp', action='store_true', help="AMP [Automatic Mixed Precision] の使用有効化")
    parser.add_argument('--opt_level', choices=['O0','O1','O2','O3'], default='O1', help='mixed precision calculation mode')
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
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


    # 各種出力ディレクトリ
    if not( os.path.exists(args.tensorboard_dir) ):
        os.mkdir(args.tensorboard_dir)
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )
    if not( os.path.exists("_debug") and args.debug ):
        os.mkdir( "_debug" )

    # for visualation
    board_train = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
    board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_test") )

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
    pass

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    pass

    #======================================================================
    # optimizer の設定
    #======================================================================
    pass

    #======================================================================
    # loss 関数の設定
    #======================================================================
    pass

    #======================================================================
    # モデルの学習処理
    #======================================================================
    pass
