import os
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import warnings
import json
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from kaggle.api.kaggle_api_extended import KaggleApi

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 自作モジュール
from dataset2 import ImaterialistDataset, ImaterialistDataLoader
from models import UNetFGVC6
from models import ParsingCrossEntropyLoss, LovaszSoftmaxLoss, VGGLoss, VanillaGANLoss, LSGANLoss, HingeGANLoss, ConditionalExpressionLoss
from utils import save_checkpoint, load_checkpoint, convert_rle
from utils import board_add_image, board_add_images, save_image_w_norm
from utils import iou_metric, iou_metric_batch
from utils import split_masks, concat_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="single_model", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="imaterialist-fashion-2019-FGVC6")
    parser.add_argument("--train_mode", choices=["train", "test", "eval"], default="train", help="")
    parser.add_argument("--model_type_G", choices=["unet_fgvc6"], default="unet4", help="生成器モデルの種類")    
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path_G', type=str, default="", help="生成器モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=100, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=4, help="バッチサイズ")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=0.001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_height', type=int, default=256, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=192, help="入力画像の幅（pixel単位）")
    parser.add_argument("--n_channels", type=int, default=3, help="チャンネル数") 
    parser.add_argument("--n_classes", type=int, default=47, help="ラベル数")   
    parser.add_argument("--n_samplings", type=int, default=100000, help="ラベル数")
    parser.add_argument('--data_augument', action='store_true')

    parser.add_argument('--lambda_l1', type=float, default=5.0, help="L1損失関数の係数値")
    parser.add_argument('--lambda_vgg', type=float, default=5.0, help="VGG perceptual loss_G の係数値")
    parser.add_argument('--lambda_bce', type=float, default=1.0, help="クロスエントロピー損失関数の係数値")
    parser.add_argument('--lambda_entropy', type=float, default=1.0, help="クロスエントロピー損失関数の係数値")
    parser.add_argument('--lambda_parsing_entropy', type=float, default=1.0, help="クロスエントロピー損失関数の係数値")

    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=10,)

    parser.add_argument("--val_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 実験名を自動的に変更
    if( args.exper_name == "single_model" ):
        if( args.train_mode in ["test", "eval"] ):
            args.exper_name = "test_" + args.exper_name
        args.exper_name += "_" + args.model_type_G
        if( args.data_augument ):
            args.exper_name += "_da"

        args.exper_name += "_ep" + str(args.n_epoches)
        args.exper_name += "_b" + str(args.batch_size)
        args.exper_name += "_lr{}".format(args.lr)
        args.exper_name += "_enpropy{}".format(args.lambda_entropy)
        args.exper_name += "_l1{}".format(args.lambda_l1)
        args.exper_name += "_vgg{}".format(args.lambda_vgg)

    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))
        if not os.path.isdir("_debug"):
            os.mkdir("_debug")

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name, "test") ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name, "test"))
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name, "test", "images") ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name, "test", "images"))
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name, "test", "masks") ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name, "test", "masks"))
    if( args.train_mode in ["train"] ):
        if not( os.path.exists(args.save_checkpoints_dir) ):
            os.mkdir(args.save_checkpoints_dir)
        if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
            os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )

    # 警告非表示
    warnings.simplefilter('ignore', DeprecationWarning)

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

    # for visualation
    if( args.train_mode == "train" ):
        board_train = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
        board_valid = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_valid") )

    board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_test") )

    #================================
    # データセットの読み込み
    #================================    
    df_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )

    # 学習用データセットとテスト用データセットの設定
    ds_train = ImaterialistDataset( args, args.dataset_dir, datamode = "train", image_height = args.image_height, image_width = args.image_width, n_classes = args.n_classes, data_augument = args.data_augument, debug = args.debug )
    ds_test = ImaterialistDataset( args, args.dataset_dir, datamode = "test", image_height = args.image_height, image_width = args.image_width, n_classes = args.n_classes, data_augument = False, debug = args.debug )

    dloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers = args.n_workers, pin_memory = True )
    dloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size_test, shuffle=False, num_workers = args.n_workers, pin_memory = True )
    
    #================================
    # 前処理
    #================================
    index = np.arange(len(ds_train))
    train_index, valid_index = train_test_split( index, test_size=args.val_rate, random_state=args.seed )
    if( args.debug ):
        print( "train_index.shape : ", train_index.shape )
        print( "valid_index.shape : ", valid_index.shape )
        print( "train_index[0:10] : ", train_index[0:10] )
        print( "valid_index[0:10] : ", valid_index[0:10] )

    dloader_train = torch.utils.data.DataLoader(Subset(ds_train, train_index), batch_size=args.batch_size, shuffle=True, num_workers = args.n_workers, pin_memory = True )
    dloader_valid = torch.utils.data.DataLoader(Subset(ds_train, valid_index), batch_size=args.batch_size_valid, shuffle=False, num_workers = args.n_workers, pin_memory = True )

    #================================
    # モデルの構造を定義する。
    #================================
    # 生成器
    if( args.model_type_G == "unet_fgvc6" ):
        model_G = UNetFGVC6( n_channels = args.n_channels, n_classes = args.n_classes ).to( device )

    if( args.debug ):
        print( "model_G :\n", model_G )

    # モデルを読み込む
    if not args.load_checkpoints_path_G == '' and os.path.exists(args.load_checkpoints_path_G):
        load_checkpoint(model_G, device, args.load_checkpoints_path_G )

    #================================
    # optimizer_G の設定
    #================================
    optimizer_G = optim.Adam(
        params = model_G.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )

    #================================
    # loss_G 関数の設定
    #================================
    loss_l1_fn = nn.L1Loss()
    loss_vgg_fn = VGGLoss(device, args.n_classes)
    loss_entropy_fn = nn.CrossEntropyLoss()
    loss_bce_fn = nn.BCEWithLogitsLoss()
    loss_entropy_fn = nn.CrossEntropyLoss()
    loss_parsing_entropy_fn = ParsingCrossEntropyLoss()

    #================================
    # モデルの学習
    #================================    
    if( args.train_mode == "train" ):
        print("Starting Training Loop...")
        n_print = 1
        step = 0
        for epoch in tqdm( range(args.n_epoches), desc = "Epoches" ):
            #=====================================
            # 学習用データの処理
            #=====================================
            for iter, inputs in enumerate( tqdm( dloader_train, desc = "minbatch iters" ) ):
                model_G.train()            

                # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                if inputs["image"].shape[0] != args.batch_size:
                    break

                # ミニバッチデータを GPU へ転送
                image_name = inputs["image_name"]
                image = inputs["image"].to(device)
                mask_split_int = inputs["mask_split_int"].to(device)
                mask_split_float = inputs["mask_split_float"].to(device)
                mask_concat_int = inputs["mask_concat_int"].to(device)
                mask_concat_float = inputs["mask_concat_float"].to(device)
                if( args.debug and n_print > 0):
                    print( "image_name : ", image_name )
                    print( "image.shape : ", image.shape )
                    print( "mask_split_int.shape : ", mask_split_int.shape )
                    print( "mask_split_float.shape : ", mask_split_float.shape )
                    print( "mask_concat_int.shape : ", mask_concat_int.shape )
                    print( "mask_concat_float.shape : ", mask_concat_float.shape )
                    print( "mask_split_int.dtype : ", mask_split_int.dtype )
                    print( "mask_split_float.dtype : ", mask_split_float.dtype )
                    print( "mask_concat_int.dtype : ", mask_concat_int.dtype )
                    print( "mask_concat_float.dtype : ", mask_concat_float.dtype )

                #====================================================
                # 学習処理
                #====================================================
                #----------------------------------------------------
                # 生成器 の forword 処理
                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                output, output_mask, output_none_act = model_G( image )

                output_concat_np = np.zeros( (args.batch_size, 1, args.image_height, args.image_width) )
                for batch_idx in range(args.batch_size):
                    output_concat_np[batch_idx,0,:,:] = concat_masks( output[batch_idx].detach().cpu().numpy().transpose(1,2,0), n_classes = args.n_classes )
                output_concat_float = torch.from_numpy( output_concat_np ).float()
                if( args.debug and n_print > 0 ):
                    print( "output.shape :", output.shape )
                    print( "output_mask.shape :", output_mask.shape )
                    print( "output_none_act.shape :", output_none_act.shape )
                    print( "output_concat_float.shape :", output_concat_float.shape )
                    print( "output.dtype :", output.dtype )
                    print( "output_concat_float.dtype :", output_concat_float.dtype )

                #----------------------------------------------------
                # 生成器の更新処理
                #----------------------------------------------------
                # 損失関数を計算する
                loss_l1 = loss_l1_fn( output, mask_split_float )
                loss_vgg = loss_vgg_fn( output, mask_split_float )
                loss_bce = loss_bce_fn( output, mask_split_float )
                loss_entropy = loss_entropy_fn( output_none_act, mask_concat_int )
                loss_parsing_entropy = loss_parsing_entropy_fn( output, mask_split_float )
                loss_G = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_entropy * loss_entropy + args.lambda_parsing_entropy * loss_parsing_entropy + args.lambda_bce * loss_bce

                # ネットワークの更新処理
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                #====================================================
                # 学習過程の表示
                #====================================================
                if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                    board_train.add_scalar('G/loss_G', loss_G.item(), step)
                    board_train.add_scalar('G/loss_l1', loss_l1.item(), step)
                    board_train.add_scalar('G/loss_vgg', loss_vgg.item(), step)
                    board_train.add_scalar('G/loss_bce', loss_bce.item(), step)
                    board_train.add_scalar('G/loss_entropy', loss_entropy.item(), step)
                    board_train.add_scalar('G/loss_parsing_entropy', loss_parsing_entropy.item(), step)
                    print( "step={}, loss_G={:.5f}, loss_l1={:.5f}, loss_vgg={:.5f}, loss_bce={:.5f}, loss_entropy={:.5f}, loss_parsing_entropy={:.5f}".format(step, loss_G, loss_l1, loss_vgg, loss_bce, loss_entropy, loss_parsing_entropy) )

                    visuals = [
                        [ image, mask_concat_float, output_concat_float ],
                    ]
                    """
                    visuals = [
                        [ image, mask_concat_float, output_concat_float ],
                        [ mask_split_float[:,i,:,:] for i in range(0,5) ],
                        [ mask_split_float[:,i,:,:] for i in range(6,10) ],
                        [ mask_split_float[:,i,:,:] for i in range(11,15) ],
                        [ mask_split_float[:,i,:,:] for i in range(16,20) ],
                        [ mask_split_float[:,i,:,:] for i in range(21,25) ],
                        [ mask_split_float[:,i,:,:] for i in range(26,30) ],
                        [ mask_split_float[:,i,:,:] for i in range(40,args.n_classes) ],
                        [ output[:,i,:,:] for i in range(0,5) ],
                        [ output[:,i,:,:] for i in range(6,10) ],
                        [ output[:,i,:,:] for i in range(11,15) ],
                        [ output[:,i,:,:] for i in range(16,20) ],
                        [ output[:,i,:,:] for i in range(21,25) ],
                        [ output[:,i,:,:] for i in range(26,30) ],
                        [ output[:,i,:,:] for i in range(40,args.n_classes) ],
                    ]
                    """

                    if( args.debug and n_print > 0 ):
                        for col, vis_item_row in enumerate(visuals):
                            for row, vis_item in enumerate(vis_item_row):
                                print("[train] vis_item[{}][{}].shape={} :".format(row,col,vis_item.shape) )

                    board_add_images(board_train, 'train', visuals, step+1)

                #=====================================
                # 検証用データの処理
                #=====================================
                if( step == 0 or ( step % args.n_display_valid_step == 0 ) ):
                    loss_G_total = 0
                    loss_l1_total, loss_vgg_total = 0, 0
                    loss_bce_total, loss_entropy_total, loss_parsing_entropy_total = 0, 0, 0
                    n_valid_loop = 0
                    for iter, inputs in enumerate( tqdm(dloader_valid) ):
                        model_G.eval()            

                        # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                        if inputs["image"].shape[0] != args.batch_size_valid:
                            break

                        # ミニバッチデータを GPU へ転送
                        image_name = inputs["image_name"]
                        image = inputs["image"].to(device)
                        mask_split_int = inputs["mask_split_int"].to(device)
                        mask_split_float = inputs["mask_split_float"].to(device)
                        mask_concat_int = inputs["mask_concat_int"].to(device)
                        mask_concat_float = inputs["mask_concat_float"].to(device)

                        #====================================================
                        # 推論処理
                        #====================================================
                        # 生成器
                        with torch.no_grad():
                            output, output_mask, output_none_act = model_G( image )
                            output_concat_np = np.zeros( (args.batch_size_valid, 1, args.image_height, args.image_width) )
                            for batch_idx in range(args.batch_size_valid):
                                output_concat_np[batch_idx,0,:,:] = concat_masks( output[batch_idx].detach().cpu().numpy().transpose(1,2,0), n_classes = args.n_classes )
                            output_concat_float = torch.from_numpy( output_concat_np ).float()

                        #----------------------------------------------------
                        # 損失関数の計算
                        #----------------------------------------------------
                        # 生成器
                        loss_l1 = loss_l1_fn( output, mask_split_float )
                        loss_vgg = loss_vgg_fn( output, mask_split_float )
                        loss_bce = loss_bce_fn( output, mask_split_float )
                        loss_entropy = loss_entropy_fn( output_none_act, mask_concat_int )
                        loss_parsing_entropy = loss_parsing_entropy_fn( output, mask_split_float )
                        loss_G = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_entropy * loss_entropy + args.lambda_parsing_entropy * loss_parsing_entropy + args.lambda_bce * loss_bce

                        # total
                        loss_G_total += loss_G
                        loss_l1_total += loss_l1
                        loss_vgg_total += loss_vgg
                        loss_bce_total += loss_bce
                        loss_entropy_total += loss_entropy
                        loss_parsing_entropy_total += loss_parsing_entropy

                        # 
                        if( iter <= args.batch_size ):
                            visuals = [
                                [ image, mask_concat_float, output_concat_float ],
                            ]

                            board_add_images(board_valid, 'valid/{}'.format(iter), visuals, step+1)

                        n_valid_loop += 1

                    #----------------------------------------------------
                    # 表示処理
                    #----------------------------------------------------
                    # 生成器
                    board_valid.add_scalar('G/loss_G', loss_G_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_l1', loss_l1_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_vgg', loss_vgg_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_bce', loss_bce_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_entropy', loss_entropy_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_parsing_entropy', loss_parsing_entropy_total.item()/n_valid_loop, step)

                step += 1
                n_print -= 1

            #====================================================
            # モデルの保存
            #====================================================
            if( epoch % args.n_save_epoches == 0 ):
                save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_ep%03d.pth' % (epoch)) )
                save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth') )
                print( "saved checkpoints" )
                
        save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth') )
        print("Finished Training Loop.")

    #================================
    # テスト用データでの推論処理
    #================================
    print("Starting eval Test Loop...")
    y_pred_test = []
    test_image_names = []
    model_G.eval()
    for step, inputs in enumerate( tqdm( dloader_test, desc = "Samplings" ) ):
        if inputs["image"].shape[0] != args.batch_size_test:
            break

        image_name = inputs["image_name"]
        test_image_names.append(image_name[0])
        image = inputs["image"].to(device)

        # 生成器 G の 推論処理
        with torch.no_grad():
            #output = model_G( image )
            output, output_mask, output_none_act = model_G( image )
            y_pred_test.append( ( (output_mask[0].detach().cpu().numpy()) * 255 ).astype('uint8') )
            #y_pred_test.append( ( (output[0].detach().cpu().numpy() + 1.0) * 0.5 * 255 ).astype('uint8') )

        n_display_images = 50
        if( step <= n_display_images ):
            visuals = [
                [image, output],
            ]
            board_add_images(board_test, 'test/{}'.format(step), visuals, -1 )

            save_image_w_norm( image, os.path.join( args.results_dir, args.exper_name, "test", "images", image_name[0] ) )
            save_image_w_norm( output, os.path.join( args.results_dir, args.exper_name, "test", "masks", image_name[0] ) )

        if( step >= args.n_samplings ):
            break

    y_pred_test = np.array( y_pred_test )
    if( args.debug ):
        print( "type(y_pred_test) : ", type(y_pred_test) )
        print( "y_pred_test.shape : ", y_pred_test.shape )

    #================================
    # Kaggle API での submit
    #================================
    # RLE [Run Length Encoding] 形式で提出のため生成画像を元の画像サイズに変換
    image_names = []
    encoded_pixels = []
    class_ids = []
    for i,name in enumerate(test_image_names):
        # 複数のマスク画像に分割
        test_mask_splits, class_ids = split_masks( mask_np = y_pred_test[i,0,:,:].squeeze(), n_classes = args.n_classes, threshold = 0 )
        print( "name={}, len(test_mask_splits)={}, class_ids={}".format(name, len(test_mask_splits), class_ids) )
        for j, label in enumerate(class_ids):
            # 元の解像度への resize
            image_height_org = ds_test.df_test.loc[name]["Height"]
            image_width_org = ds_test.df_test.loc[name]["Width"]
            #print( "image_height_org={}, image_width_org={}".format(image_height_org, image_width_org) )
            #print( "test_mask_splits[{}].shape : {}".format(j,test_mask_splits[j].shape) )
            y_pred_test_org = cv2.resize( test_mask_splits[j], (image_height_org, image_width_org), interpolation = cv2.INTER_NEAREST )

            # 画像ファイル名
            image_names.append(name)

            # RLE の計算
            encoded_pixels.append( convert_rle(np.round(y_pred_test_org)) )

            # クラスラベル
            class_ids.append(label)

            # ラベル別画像の保存
            cv2.imwrite( os.path.join( args.results_dir, args.exper_name, "test", "masks", name.split(".png")[0] + "_{}.png".format(label) ), y_pred_test_org ) 
            if( j >= len(test_mask_splits) - 1):
                break

    df_submission = pd.DataFrame( {'EncodedPixels' : encoded_pixels, 'ClassId' : class_ids}, index = image_names )
    df_submission.index.names = ['ImageId']
    df_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file) )

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
    