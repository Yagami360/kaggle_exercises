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
#from skimage.transform import resize
from kaggle.api.kaggle_api_extended import KaggleApi

# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 自作モジュール
from dataset import load_dataset, TGSSaltDataset, TGSSaltDataLoader
from models import UNet4, UNet4BottleNeck, GANimationGenerator, MGVTONResGenerator
from models import PatchGANDiscriminator, MultiscaleDiscriminator, GANimationDiscriminator
from models import ParsingCrossEntropyLoss, VGGLoss, VanillaGANLoss, LSGANLoss, HingeGANLoss, ConditionalExpressionLoss
from utils import save_checkpoint, load_checkpoint, convert_rle
from utils import board_add_image, board_add_images, save_image_w_norm
from utils import iou_metric, iou_metric_batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="single_model_pytorch", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="../datasets/competition_data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="tgs-salt-identification-challenge")
    parser.add_argument("--train_mode", choices=["train", "test", "eval"], default="train", help="")
    parser.add_argument("--model_type_G", choices=["unet4", "unet5", "unet4bottleneck", "mgvton", "ganimation"], default="unet4", help="生成器モデルの種類")
    parser.add_argument("--model_type_D", choices=["patchgan", "multiscale", "ganimation"], default="patchgan", help="識別器モデルの種類")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path_G', type=str, default="", help="生成器モデルの読み込みファイルのパス")
    parser.add_argument('--load_checkpoints_path_D', type=str, default="", help="識別器モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=200, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=32, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=0.001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_height_org', type=int, default=101, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width_org', type=int, default=101, help="入力画像の幅（pixel単位）")
    parser.add_argument('--image_height', type=int, default=128, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=128, help="入力画像の幅（pixel単位）")
    parser.add_argument("--n_channels", type=int, default=1, help="チャンネル数")    
    parser.add_argument("--n_samplings", type=int, default=100000, help="ラベル数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument('--depth', action='store_true')
    parser.add_argument("--val_rate", type=float, default=0.20)

    parser.add_argument('--lambda_bce', type=float, default=1.0, help="クロスエントロピー損失関数の係数値")
    parser.add_argument('--lambda_enpropy', type=float, default=1.0, help="クロスエントロピー損失関数の係数値")
    parser.add_argument('--lambda_l1', type=float, default=0.0, help="L1損失関数の係数値")
    parser.add_argument('--lambda_vgg', type=float, default=0.0, help="VGG perceptual loss_G の係数値")
    parser.add_argument('--lambda_adv', type=float, default=1.0, help="Adv loss_G の係数値")
    parser.add_argument('--adv_loss_type', choices=['vanilla', 'lsgan', 'hinge'], default="lsgan", help="GAN Adv loss の種類")
    parser.add_argument('--lambda_cond', type=float, default=1000.0, help="conditional expression loss の係数値")

    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument("--n_save_epoches", type=int, default=50,)

    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if( args.model_type_G == "unet4bottleneck" ):
        args.depth = True
    if( args.model_type_D == "ganimation" ):
        args.depth = True

    # 実験名を自動的に変更
    if( args.exper_name == "single_model_pytorch" ):
        if( args.train_mode in ["test", "eval"] ):
            args.exper_name = "test_" + args.exper_name
        args.exper_name += "_" + args.model_type_G
        if( args.data_augument ):
            args.exper_name += "_da"
        if( args.depth ):
            args.exper_name += "_depth"

        args.exper_name += "_ep" + str(args.n_epoches)
        args.exper_name += "_b" + str(args.batch_size)
        args.exper_name += "_lr{}".format(args.lr)
        args.exper_name += "_bce{}".format(args.lambda_bce)
        args.exper_name += "_enpropy{}".format(args.lambda_enpropy)
        args.exper_name += "_l1{}".format(args.lambda_l1)
        args.exper_name += "_vgg{}".format(args.lambda_vgg)
        args.exper_name += "_adv{}_{}".format(args.adv_loss_type, args.lambda_adv)
        if( args.model_type_D == "ganimation" ):
            args.exper_name += "_cond{}".format(args.lambda_cond)

    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name, "valid") ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name, "valid"))
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name, "valid", "images") ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name, "valid", "images"))
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name, "valid", "masks") ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name, "valid", "masks"))
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
        board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_test") )

    #================================
    # データセットの読み込み
    #================================    
    df_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )

    # 学習用データセットとテスト用データセットの設定
    ds_train = TGSSaltDataset( args, args.dataset_dir, datamode = "train", data_augument = args.data_augument, debug = args.debug )
    ds_test = TGSSaltDataset( args, args.dataset_dir, datamode = "test", data_augument = False, debug = args.debug )

    #dloader_train = TGSSaltDataLoader(ds_train, batch_size=args.batch_size, shuffle=True, n_workers=args.n_workers )
    #dloader_test = TGSSaltDataLoader(ds_test, batch_size=args.batch_size_test, shuffle=False, n_workers=args.n_workers )
    dloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers = args.n_workers, pin_memory = True )
    dloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size_test, shuffle=False, num_workers = args.n_workers, pin_memory = True )

    """
    X_train_img, y_train_mask, X_test_img, X_train_depth, X_test_depth, train_image_names, test_image_names \
    = load_dataset( 
        dataset_dir = args.dataset_dir, 
        image_height_org = args.image_height, image_width_org = args.image_width_org, 
        image_height = args.image_height, image_width = args.image_width, n_channels = args.n_channels, 
        n_samplings = args.n_samplings,
        debug = args.debug,
    )
    y_pred_train = np.zeros( y_train_mask.shape )
    if( args.debug ):
        print( "X_train_img.shape : ", X_train_img.shape )
        print( "y_train_mask.shape : ", y_train_mask.shape )
        print( "X_test_img.shape : ", X_test_img.shape )
        print( "X_train_depth.shape : ", X_train_depth.shape )
        print( "X_test_depth.shape : ", X_test_depth.shape )
    """

    # 可視化
    """
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for idx in range(max_images):
        img = X_train_img[idx]       # 0.0f ~ 1.0f
        mask = y_train_mask[idx]     # 0.0f ~ 1.0f
        ax = axs[int(idx / grid_width), idx % grid_width]
        ax.imshow(img.squeeze(), cmap="Greys")
        ax.imshow(mask.squeeze(), alpha=0.3, cmap="Greens")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.suptitle("images and masks [train]\nGreen: salt")
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "images_and_masks_train.png"), dpi = 300, bbox_inches = 'tight' )
    """

    #================================
    # 前処理
    #================================
    """
    # データセットの分割
    X_train_img, X_valid_img, y_train_mask, y_valid_mask, train_image_names, image_names_valid, X_train_depth, X_valid_depth \
    = train_test_split( X_train_img, y_train_mask, train_image_names, X_train_depth, test_size=args.val_rate, random_state=args.seed )
    if( args.debug ):
        print( "X_train_img.shape : ", X_train_img.shape )
        print( "X_valid_img.shape : ", X_valid_img.shape )
        print( "y_train_mask.shape : ", y_train_mask.shape )
        print( "y_valid_mask.shape : ", y_valid_mask.shape )
        print( "X_train_depth.shape : ", X_train_depth.shape )
        print( "X_valid_depth.shape : ", X_valid_depth.shape )
    """

    #================================
    # モデルの構造を定義する。
    #================================
    # 生成器
    if( args.model_type_G == "unet4" ):
        if( args.depth ):
            model_G = UNet4( n_in_channels = args.n_channels + 1, n_out_channels = args.n_channels, n_fmaps = 32,).to( device )
        else:
            model_G = UNet4( n_in_channels = args.n_channels, n_out_channels = args.n_channels, n_fmaps = 32,).to( device )
    elif( args.model_type_G == "unet4bottleneck" ):
        model_G = UNet4BottleNeck( n_in_channels = args.n_channels, n_out_channels = args.n_channels, n_fmaps = 32,).to( device )
    elif( args.model_type_G == "mgvton" ):
        if( args.depth ):
            model_G = MGVTONResGenerator( input_nc = args.n_channels + 1, output_nc = args.n_channels, padding_type='zero', affine=False ).to( device )
        else:
            model_G = MGVTONResGenerator( input_nc = args.n_channels, output_nc = args.n_channels, padding_type='zero', affine=False ).to( device )
            #model_G = MGVTONResGenerator( input_nc = args.n_channels, output_nc = args.n_channels, padding_type='reflect', affine=True ).to( device )
    elif( args.model_type_G == "ganimation" ):
        if( args.depth ):
            model_G = GANimationGenerator( input_nc = args.n_channels + 1, output_nc = args.n_channels, conv_dim = 32 ).to( device )
        else:
            model_G = GANimationGenerator( input_nc = args.n_channels, output_nc = args.n_channels, conv_dim = 32 ).to( device )

    # 識別器
    if( args.model_type_D == "patchgan" ):
        model_D = PatchGANDiscriminator( n_in_channels = args.n_channels, n_fmaps = 32 ).cuda()
    elif( args.model_type_D == "multiscale" ):
        model_D = MultiscaleDiscriminator( n_in_channels = args.n_channels, n_fmaps = 32, n_dis = 3  ).cuda()
    elif( args.model_type_D == "ganimation" ):
        model_D = GANimationDiscriminator( n_in_channels = args.n_channels, n_fmaps = 32, feat_dim = 1 ).cuda()

    if( args.debug ):
        print( "model_G :\n", model_G )
        print( "model_D :\n", model_D )

    # モデルを読み込む
    if not args.load_checkpoints_path_G == '' and os.path.exists(args.load_checkpoints_path_G):
        load_checkpoint(model_G, device, args.load_checkpoints_path_G )
    if not args.load_checkpoints_path_D == '' and os.path.exists(args.load_checkpoints_path_D):
        load_checkpoint(model_D, device, args.load_checkpoints_path_D )

    #================================
    # optimizer_G の設定
    #================================
    optimizer_G = optim.Adam(
        params = model_G.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )

    optimizer_D = optim.Adam(
        params = model_D.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )

    #================================
    # loss_G 関数の設定
    #================================
    loss_l1_fn = nn.L1Loss()
    loss_vgg_fn = VGGLoss(device)
    loss_entropy_fn = ParsingCrossEntropyLoss()
    loss_bce_fn = nn.BCEWithLogitsLoss()

    if( args.adv_loss_type == "vanilla" ):
        loss_adv_fn = VanillaGANLoss()
    elif( args.adv_loss_type == "lsgan" ):
        loss_adv_fn = LSGANLoss()
    elif( args.adv_loss_type == "hinge" ):
        loss_adv_fn = HingeGANLoss()
    else:
        loss_adv_fn = LSGANLoss()

    if( args.model_type_D == "ganimation" ):
        loss_cond_fn = ConditionalExpressionLoss()

    #================================
    # モデルの学習
    #================================    
    if( args.train_mode == "train" ):
        print("Starting Training Loop...")
        n_print = 1
        step = 0
        for epoch in tqdm( range(args.n_epoches), desc = "Epoches" ):
            # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
            for iter, inputs in enumerate( tqdm( dloader_train, desc = "minbatch iters" ) ):
                model_G.train()            
                model_D.train()            

                # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                if inputs["image"].shape[0] != args.batch_size:
                    break

                # ミニバッチデータを GPU へ転送
                image_name = inputs["image_name"]
                image = inputs["image"].to(device)
                mask = inputs["mask"].to(device)
                depth = inputs["depth"].to(device)
                if( args.debug and n_print > 0):
                    print( "image.shape : ", image.shape )
                    print( "mask.shape : ", mask.shape )
                    print( "depth.shape : ", depth.shape )

                #====================================================
                # 学習処理
                #====================================================
                #----------------------------------------------------
                # 生成器 の forword 処理
                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                if( args.model_type_G in ["unet4bottleneck"] ):
                        output = model_G( image, depth )
                else:
                    if( args.depth ):
                        depth = depth.expand(depth.shape[0], depth.shape[1], image.shape[2], image.shape[3] )
                        concat = torch.cat( [image, depth], dim=1)
                        output = model_G( concat )
                    else:
                        output = model_G( image )

                if( args.debug and n_print > 0 ):
                    print( "output.shape :", output.shape )

                #----------------------------------------------------
                # 識別器の更新処理
                #----------------------------------------------------
                # 無効化していた識別器 D のネットワークの勾配計算を有効化。
                for param in model_D.parameters():
                    param.requires_grad = True

                # 学習用データをモデルに流し込む
                if( args.model_type_D == "ganimation" ):
                    d_real, d_real_depth = model_D( output )
                    d_fake, d_fake_depth = model_D( output.detach() )
                    if( args.debug and n_print > 0 ):
                        print( "d_real.shape :", d_real.shape )
                        print( "d_fake.shape :", d_fake.shape )
                        print( "d_real_depth.shape :", d_real_depth.shape )
                        print( "d_fake_depth.shape :", d_fake_depth.shape )
                else:
                    d_real = model_D( output )
                    d_fake = model_D( output.detach() )
                    if( args.debug and n_print > 0 ):
                        print( "d_real.shape :", d_real.shape )
                        print( "d_fake.shape :", d_fake.shape )

                # 損失関数を計算する
                loss_D, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )
                if( args.model_type_D == "ganimation" ):
                    loss_D_cond_depth = loss_cond_fn( d_real_depth, depth[:,:,0,0] ) + loss_cond_fn( d_fake_depth, depth[:,:,0,0] )
                    loss_D = loss_D + args.lambda_cond * loss_D_cond_depth

                # ネットワークの更新処理
                optimizer_D.zero_grad()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                # 無効化していた識別器 D のネットワークの勾配計算を有効化。
                for param in model_D.parameters():
                    param.requires_grad = False

                #----------------------------------------------------
                # 生成器の更新処理
                #----------------------------------------------------
                # 損失関数を計算する
                loss_l1 = loss_l1_fn( output, mask )
                if( args.n_channels == 3 ):
                    loss_vgg = loss_vgg_fn( output, mask )
                loss_entropy = loss_entropy_fn( output, mask )
                loss_bce = loss_bce_fn( output, mask )
                loss_adv = loss_adv_fn.forward_G( d_fake )

                if( args.n_channels == 3 ):
                    loss_G = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_enpropy * loss_entropy + args.lambda_bce * loss_bce + args.lambda_adv * loss_adv
                else:
                    loss_G = args.lambda_l1 * loss_l1 + args.lambda_enpropy * loss_entropy + args.lambda_bce * loss_bce + args.lambda_adv * loss_adv

                if( args.model_type_D == "ganimation" ):
                    loss_G_cond_depth = loss_cond_fn( d_real_depth, depth[:,:,0,0] ) + loss_cond_fn( d_fake_depth, depth[:,:,0,0] )
                    loss_G = loss_G + args.lambda_cond * loss_G_cond_depth

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
                    if( args.n_channels == 3 ):
                        board_train.add_scalar('G/loss_vgg', loss_vgg.item(), step)
                    board_train.add_scalar('G/loss_entropy', loss_entropy.item(), step)
                    board_train.add_scalar('G/loss_bce', loss_bce.item(), step)
                    board_train.add_scalar('G/loss_adv', loss_adv.item(), step)
                    if( args.model_type_D == "ganimation" ):
                        board_train.add_scalar('G/loss_G_cond_depth', loss_G_cond_depth.item(), step)

                    board_train.add_scalar('D/loss_D', loss_D.item(), step)
                    board_train.add_scalar('D/loss_D_real', loss_D_real.item(), step)
                    board_train.add_scalar('D/loss_D_fake', loss_D_fake.item(), step)
                    if( args.model_type_D == "ganimation" ):
                        board_train.add_scalar('D/loss_D_cond_depth', loss_D_cond_depth.item(), step)

                    if( args.n_channels == 3 ):
                        print( "step={}, loss_G={:.5f}, loss_l1={:.5f}, loss_vgg={:.5f}, loss_entropy={:.5f}, loss_bce={:.5f}, loss_adv={:.5f}".format(step, loss_G, loss_l1, loss_vgg, loss_entropy, loss_bce, loss_adv) )
                        print( "step={}, loss_D={:.5f}, loss_D_real={:.5f}, loss_D_fake={:.5f}".format(step, loss_D.item(), loss_D_real.item(), loss_D_fake.item()) )
                        if( args.model_type_D == "ganimation" ):
                            print( "step={}, loss_G_cond_depth={:.5f}".format(step, loss_G_cond_depth,) )
                            print( "step={}, loss_D_cond_depth={:.5f}".format(step, loss_D_cond_depth,) )

                    else:
                        print( "step={}, loss_G={:.5f}, loss_l1={:.5f}, loss_entropy={:.5f}, loss_bce={:.5f}, loss_adv={:.5f}".format(step, loss_G, loss_l1, loss_entropy, loss_bce, loss_adv) )
                        print( "step={}, loss_D={:.5f}, loss_D_real={:.5f}, loss_D_fake={:.5f}".format(step, loss_D.item(), loss_D_real.item(), loss_D_fake.item()) )
                        if( args.model_type_D == "ganimation" ):
                            print( "step={}, loss_G_cond_depth={:.5f}".format(step, loss_G_cond_depth,) )
                            print( "step={}, loss_D_cond_depth={:.5f}".format(step, loss_D_cond_depth,) )

                    visuals = [
                        [image, mask, output],
                    ]
                    board_add_images(board_train, 'images', visuals, step+1)

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
    # モデルの推論処理
    #================================
    print("Starting Test Loop...")
    n_print = 1
    y_pred_test = []
    test_image_names = []
    model_G.eval()
    for step, inputs in enumerate( tqdm( dloader_test, desc = "Samplings" ) ):
        if inputs["image"].shape[0] != args.batch_size_test:
            break

        image_name = inputs["image_name"]
        test_image_names.append(image_name[0])
        image = inputs["image"].to(device)
        depth = inputs["depth"].to(device)

        # 生成器 G の 推論処理
        with torch.no_grad():
            if( args.model_type_G in ["unet4bottleneck"] ):
                    output = model_G( image, depth )
            else:
                if( args.depth ):
                    depth = depth.expand(depth.shape[0], depth.shape[1], image.shape[2], image.shape[3] )
                    concat = torch.cat( [image, depth], dim=1)
                    output = model_G( concat )
                else:
                    output = model_G( image )

            y_pred_test.append( output[0].detach().cpu().numpy() )
            if( args.debug and n_print > 0 ):
                print( "output.shape :", output.shape )
                print( "type(output) :", type(output) )

        if( step <= 10 ):
            save_image_w_norm( image, os.path.join( args.results_dir, args.exper_name, "test", "images", image_name[0] ) )
            save_image_w_norm( output, os.path.join( args.results_dir, args.exper_name, "test", "masks", image_name[0] ) )

        if( step >= args.n_samplings ):
            break

        n_print -= 1

    y_pred_test = np.array( y_pred_test )
    print( "type(y_pred_test) : ", type(y_pred_test) )
    print( "y_pred_test.shape : ", y_pred_test.shape )

    #================================
    # 可視化処理
    #================================
    # IoU
    """
    thresholds = np.linspace(0, 1, 50)  # IoU スコアの低い結果を除外するためのスレッショルド
    ious = np.array( [iou_metric_batch(y_valid_mask, np.int32(y_pred_train > threshold)) for threshold in thresholds] )

    threshold_best_index = np.argmax(ious[9:-10]) + 9
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]   # ?
    print( "iou_best = {:0.4f} ".format(iou_best) )
    print( "threshold_best = {:0.4f} ".format(threshold_best) )

    fig, axs = plt.subplots()
    axs.plot(thresholds, ious)
    axs.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.grid()
    plt.legend()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "IoU.png"), dpi = 300, bbox_inches = 'tight' )
    """

    # 元画像と生成マスク画像の重ね合わせ（test）
    """
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for i, name in enumerate(test_image_names[0:max_images]):
        img = X_test_img[i]
        mask = np.array(np.round(y_pred_test[i] > threshold_best), dtype=np.float32)
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img.squeeze(), cmap="Greys")
        ax.imshow(mask.squeeze(), alpha=0.3, cmap="Greens")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.suptitle("images and masks [test]\nGreen: salt")
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "images_and_masks_test.png"), dpi = 300, bbox_inches = 'tight' )
    """

    # 元画像と生成マスク画像の重ね合わせ（valid）
    """
    max_images = 60
    grid_width = 15
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for i, name in enumerate(image_names_valid[0:max_images]):
        img = X_valid_img[i]
        mask = y_valid_mask[i]
        pred_mask = np.array(np.round(y_pred_train[i] > threshold_best), dtype=np.float32)
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img.squeeze(), cmap="Greys")
        ax.imshow(mask.squeeze(), alpha=0.3, cmap="Greens")
        ax.imshow(pred_mask.squeeze(), alpha=0.3, cmap="OrRd")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    plt.suptitle("images and masks [train]\nGreen: salt [correct], Red: salt [predict]")
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "images_and_masks_valid.png"), dpi = 300, bbox_inches = 'tight' )
    """

    #================================
    # Kaggle API での submit
    #================================
    # RLE [Run Length Encoding] 形式で提出のため生成画像を元の画像サイズに変換
    y_pred_test_org = np.zeros( (len(y_pred_test), args.image_height_org, args.image_width_org), dtype=np.float32 )
    for i in range(len(y_pred_test)):
        y_pred_test_org[i] = cv2.resize( y_pred_test[i,0,:,:].squeeze(), (args.image_height_org, args.image_width_org), interpolation = cv2.INTER_NEAREST )
        #y_pred_test_org[i] = resize( y_pred_test[i,0,:,:].squeeze(), (args.image_height_org, args.image_width_org), mode='constant', preserve_range=True )

    # 提出用データに値を設定
    y_sub = { name.split(".png")[0] : convert_rle(np.round(y_pred_test_org[i] > 0.0)) for i,name in enumerate(test_image_names) }
    df_submission = pd.DataFrame.from_dict( y_sub, orient='index' )
    df_submission.index.names = ['id']
    df_submission.columns = ['rle_mask']
    df_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file) )

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
    