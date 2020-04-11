import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random
import warnings
import pandas as pd
from matplotlib import pyplot as plt
#from apex import amp
from kaggle.api.kaggle_api_extended import KaggleApi

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# scikit-learn
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# 自作クラス
from dataset import DogsVSCatsDataset, DogsVSCatsDataLoader
from networks import MyResNet18, ResNet18, ResNet50
from utils import save_checkpoint, load_checkpoint
from utils import board_add_image, board_add_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="dog-vs-cat_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset_dir', type=str, default="datasets", help="データセットのディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--network_type', choices=['my_resnet18', 'resnet18', 'resnet50'], default="resnet50", help="ネットワークの種類")
    parser.add_argument('--pretrained', action='store_true', help="事前学習済みモデルを行うか否か")
    parser.add_argument('--n_steps', type=int, default=10000, help="学習ステップ数")
    parser.add_argument("--n_splits", type=int, default=4, help="k-fold CV での学習用データセットの分割数")
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=4, help="test データのバッチサイズ")
    parser.add_argument('--lr', type=float, default=0.0001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")

    parser.add_argument('--image_height', type=int, default=224, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=224, help="入力画像の幅（pixel単位）")
    parser.add_argument('--n_fmaps', type=int, default=64, help="１層目の特徴マップの枚数")
    parser.add_argument('--enable_da', action='store_true', help="Data Augumentation を行うか否か")

    parser.add_argument('--n_display_step', type=int, default=50, help="tensorboard への表示間隔")
    parser.add_argument('--n_display_test_step', type=int, default=500, help="test データの tensorboard への表示間隔")
    parser.add_argument("--n_save_step", type=int, default=1000, help="モデルのチェックポイントの保存間隔")
    parser.add_argument("--seed", type=int, default=71, help="乱数シード値")
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
    if ( os.path.exists("_debug") == False and args.debug ):
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
    ds_train = DogsVSCatsDataset( args, args.dataset_dir, datamode = "train", enable_da = args.enable_da )
    dloader_train = DogsVSCatsDataLoader(ds_train, batch_size=args.batch_size, shuffle=True, n_workers=args.n_workers )

    #======================================================================
    # loss 関数の設定
    #======================================================================
    loss_fn = nn.CrossEntropyLoss()

    #======================================================================
    # k-fold CV でのモデルの学習処理
    #======================================================================
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = KFold( n_splits=args.n_splits, shuffle=True, random_state=args.seed )
    #kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for fold_id, (train_index, valid_index) in enumerate( kf.split(ds_train.image_names, ds_train.image_names) ):
        print( "fold_id={}, train_index[0:10]={}, valid_index[0:10]={}".format(fold_id, train_index[0:10], valid_index[0:10]) )
        print( "len(train_index)={}, len(valid_index)={}".format(len(train_index), len(valid_index)) )

        #======================================================================
        # モデルの構造を定義する。
        #======================================================================
        if( args.network_type == "my_resnet18" ):
            model = MyResNet18( n_in_channels = 3, n_fmaps = args.n_fmaps, n_classes = 2 ).to(device)
        elif( args.network_type == "resnet18" ):
            model = ResNet18( n_classes = 2, pretrained = args.pretrained ).to(device)
        else:
            model = ResNet50( n_classes = 2, pretrained = args.pretrained ).to(device)

        # モデルを読み込む
        if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
            load_checkpoint(model, device, args.load_checkpoints_path )

        #======================================================================
        # optimizer の設定
        #======================================================================
        optimizer = optim.Adam(
            params = model.parameters(),
            lr = args.lr, betas = (args.beta1,args.beta2)
        )

        #======================================================================
        # 分割されたデータセットの作成
        #======================================================================
        dloader_train_fold = DogsVSCatsDataLoader(Subset(ds_train, train_index), batch_size=args.batch_size, shuffle=True, n_workers=args.n_workers )
        dloader_val_fold = DogsVSCatsDataLoader(Subset(ds_train, valid_index), batch_size=args.batch_size_test, shuffle=False, n_workers=args.n_workers )

        #====================================================
        # 学習処理
        #====================================================
        print("Starting Training Loop...")
        n_print = 1
        for step in tqdm( range(args.n_steps ), desc = "train steps" ):
            model.train()

            # ミニバッチデータを GPU へ転送
            inputs = dloader_train_fold.next_batch()
            image_name = inputs["image_name"]
            image = inputs["image"].to(device)
            targets = inputs["targets"].to(device)
            if( args.debug and n_print > 0):
                print( "image.shape : ", image.shape )
                print( "targets.shape : ", targets.shape )
                print( "targets : ", targets )

            #----------------------------------------------------
            # 学習用データをモデルに流し込む
            #----------------------------------------------------
            output = model( image )
            if( args.debug and n_print > 0 ):
                print( "output.shape :", output.shape )

            #----------------------------------------------------
            # 損失関数を計算する
            #----------------------------------------------------
            loss = loss_fn( output, targets )

            #----------------------------------------------------
            # ネットワークの更新処理
            #----------------------------------------------------
            # 勾配を 0 に初期化
            optimizer.zero_grad()

            # 勾配計算
            loss.backward()

            # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
            optimizer.step()

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_display_step == 0 ) ):
                board_train.add_scalar('Model/loss', loss.item(), step+1)
                print( "step={}, loss={:.5f}".format(step+1, loss) )

                board_add_image(board_train, 'input image', image, step+1, n_max_images = 4)
                #board_train.add_histogram('output[0]', output[0], step+1) 
                #print( "step={}, targets=({}), output=({:.5f},{:.5f})".format(step+1, targets[0].item(), output[0,0], output[0,1]) )

                #----------------------------------------------------
                # 正解率を計算する。（バッチデータ）
                #----------------------------------------------------
                # 確率値が最大のラベル 0~9 を予想ラベルとする。
                # dim = 1 ⇒ 列方向で最大値をとる
                # Returns : (Tensor, LongTensor)
                _, predicts = torch.max( output.data, dim = 1 )
                if( args.debug and n_print > 0 ):
                    print( "predicts.shape :", predicts.shape )

                # 正解数のカウント
                n_tests = targets.size(0)

                # ミニバッチ内で一致したラベルをカウント
                n_correct = ( predicts == targets ).sum().item()

                accuracy = n_correct / n_tests
                print( "step={}, accuracy [train batch]={:.5f}".format(step+1, accuracy) )
                board_train.add_scalar('Model/accuracy_batch', accuracy, step+1)

            #====================================================
            # val loss の表示
            #====================================================
            if( step == 0 or ( step % args.n_display_test_step == 0 ) ):
                if( len(dloader_val_fold.dataset) > dloader_val_fold.batch_size ):
                    n_val_loop = len(dloader_val_fold.dataset) // dloader_val_fold.batch_size
                else:
                    n_val_loop = len(dloader_val_fold.dataset)

                model.eval()
                loss_total = 0
                n_vals = 0
                n_correct = 0
                for i in range( n_val_loop ):
                    # 入力データをセット
                    inputs = dloader_val_fold.next_batch()
                    image_name = inputs["image_name"]
                    image = inputs["image"].to(device)
                    targets = inputs["targets"].to(device)

                    # 学習用データをモデルに流し込む
                    with torch.no_grad():
                        output = model( image )

                    # 損失関数を計算する
                    loss = loss_fn( output, targets )
                    loss_total += loss

                    # 正解率を計算する。
                    _, predicts = torch.max( output.data, dim = 1 )
                    n_vals += targets.size(0)
                    n_correct += ( predicts == targets ).sum().item()

                board_test.add_scalar('Model/loss', (loss_total/n_val_loop), step+1)

                accuracy = n_correct / n_vals
                board_test.add_scalar('Model/accuracy', accuracy, step+1)
                print( "step={}, accuracy [val]={:.5f}".format(step+1, accuracy) )

            #====================================================
            # モデルの保存
            #====================================================
            if( ( step % args.n_save_step == 0 ) ):
                save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'step_{:08}_kfold{}.pth'.format(step+1,fold_id)) )
                save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final_kfold{}.pth'.format(fold_id)) )
                print( "saved checkpoints" )
            
            n_print -= 1

        save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final_kfold{}.pth').format(fold_id) )
        print("Finished Training Loop.")

    #====================================================
    # モデルの推論処理（k-fold CV での各 k 値の学習済みモデルを平均化）
    #====================================================
    pass
