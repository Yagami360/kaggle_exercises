import os
import argparse
import numpy as np
import pandas as pd
import random
import warnings
import json
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from skimage.transform import resize
from kaggle.api.kaggle_api_extended import KaggleApi

# keras
import keras
from keras.preprocessing import image
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend.tensorflow_backend as KTF

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# 自作モジュール
from dataset import load_dataset
from models import KerasUnet, KerasUnetWithDepth
from utils import save_checkpoint, load_checkpoint, convert_rle
from utils import iou_metric, iou_metric_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="single_model_kearas", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="../datasets/competition_data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="tgs-salt-identification-challenge")
    parser.add_argument("--train_mode", choices=["train", "test", "eval"], default="train", help="")
    parser.add_argument("--model_type", choices=["unet", "unet_depth"], default="unet", help="分類器モデルの種類")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument("--n_epoches", type=int, default=200, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=32, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=0.001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_height_org', type=int, default=101, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width_org', type=int, default=101, help="入力画像の幅（pixel単位）")
    parser.add_argument('--image_height', type=int, default=128, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=128, help="入力画像の幅（pixel単位）")
    parser.add_argument("--n_channels", type=int, default=1, help="チャンネル数")    
    parser.add_argument("--n_samplings", type=int, default=-1, help="ラベル数")
    parser.add_argument('--data_augument_type', choices=["none_da", "da1", "da2"], default="da1", help="Data Augumentation の種類")
    parser.add_argument("--val_rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if( args.model_type == "unet_depth" ):
        args.data_augument_type = "none_da"

    # 実験名を自動的に変更
    if( args.exper_name == "single_model_kearas" ):
        if( args.train_mode in ["test", "eval"] ):
            args.exper_name = "test_" + args.exper_name
        args.exper_name += "_" + args.model_type
        args.exper_name += "_ep" + str(args.n_epoches)
        args.exper_name += "_b" + str(args.batch_size)
        args.exper_name += "_lr{}".format(args.lr)
        args.exper_name += "_" + args.data_augument_type

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

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)

    #================================
    # データセットの読み込み
    #================================    
    df_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )

    # 学習用データセットとテスト用データセットの設定
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

    # 可視化
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

    #================================
    # 前処理
    #================================
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

    # データオーギュメントとバッチ単位で学習のための DataGenerator
    if( args.data_augument_type == "da1" ):
        datagen = ImageDataGenerator(
            featurewise_center = False,               # set input mean to 0 over the dataset
            samplewise_center = False,                # set each sample mean to 0
            featurewise_std_normalization = False,    # divide inputs by std of the dataset
            samplewise_std_normalization = False,     # divide each input by its std
            zca_whitening = False,                    # apply ZCA whitening
            rotation_range = 10,                      # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1,                         # Randomly zoom image 
            width_shift_range = 0.1,                  # randomly shift images horizontally (fraction of total width)
            height_shift_range = 0.1,                 # randomly shift images vertically (fraction of total height)
            horizontal_flip = True,                   # randomly flip images
            vertical_flip = True,                     # randomly flip images
        )
        datagen.fit(X_train_img)
    elif( args.data_augument_type == "da2" ):
        datagen = ImageDataGenerator()
        datagen.fit(X_train_img)

        # np.fliplr() : 左右反転文を DA
        X_train_img = np.append(X_train_img, [np.fliplr(x) for x in X_train_img], axis=0)
        y_train_mask = np.append(y_train_mask, [np.fliplr(x) for x in y_train_mask], axis=0)
        #X_valid_img = np.append(X_valid_img, [np.fliplr(x) for x in X_valid_img], axis=0)
        #y_valid_mask = np.append(y_valid_mask, [np.fliplr(x) for x in y_valid_mask], axis=0)
        print( "X_train_img.shape: ", X_train_img.shape )
        print( "y_train_mask.shape: ", y_train_mask.shape )
    else:
        datagen = ImageDataGenerator()
        datagen.fit(X_train_img)

    #================================
    # モデルの学習 & 推論
    #================================    
    #--------------------
    # keras の call back
    #--------------------
    # 各エポック終了毎のモデルのチェックポイント保存用 call back
    callback_checkpoint = ModelCheckpoint( 
        filepath = os.path.join(args.save_checkpoints_dir, args.exper_name, "model_ep{epoch:03d}.hdf5" ), 
        monitor = 'loss', 
        verbose = 1, 
        save_weights_only = True,   # 
        save_best_only = False,     # 精度がよくなった時だけ保存するかどうか指定。False の場合は毎 epoch 毎保存．
        mode = 'auto',              # 
        period = 10                 # 何エポックごとに保存するか
    )
    callbacks = [ callback_checkpoint ]

    #--------------------
    # モデルの定義
    #--------------------
    if( args.model_type == "unet" ):
        model = KerasUnet(
            image_height = args.image_height, image_width = args.image_width, n_in_channels = args.n_channels,
            n_epoches = args.n_epoches, batch_size = args.batch_size, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2,
            use_valid = True, callbacks = callbacks, use_datagen = True, datagen = datagen, debug = args.debug
        )
    elif( args.model_type == "unet_depth" ):
        # unet_depth の入力形式 X_train = {'img': X_train_img, 'depth': X_train_depth } では datagen 使用不可
        model = KerasUnetWithDepth(
            image_height = args.image_height, image_width = args.image_width, n_in_channels = args.n_channels,
            n_epoches = args.n_epoches, batch_size = args.batch_size, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2,
            use_valid = True, callbacks = callbacks, use_datagen = False, datagen = datagen, debug = args.debug
        )

    # モデルを読み込む
    if( args.load_checkpoints_path != '' and os.path.exists(args.load_checkpoints_path) ):
        load_checkpoint(model.model, args.load_checkpoints_path )

    if( args.model_type == "unet_depth" ):
        X_train = {'img': X_train_img, 'depth': X_train_depth }
        X_valid = {'img': X_valid_img, 'depth': X_valid_depth }
        X_test = {'img': X_test_img, 'depth': X_test_depth }
    else:
        X_train = X_train_img
        X_valid = X_valid_img
        X_test = X_test_img

    #--------------------
    # モデルの学習処理
    #--------------------
    if( args.train_mode in ["train"] ):
        model.fit(X_train, y_train_mask, X_valid, y_valid_mask)

    elif( args.train_mode in ["eval"] ):
        eval_results_train = model.evaluate( X_train, y_train_mask )
        eval_results_val = model.evaluate( X_valid, y_valid_mask )
        print( "loss [train]={:0.5f}, accuracy [train]={:0.5f}".format(eval_results_train[0], eval_results_train[1]) )
        print( "loss [valid]={:0.5f}, accuracy [valid]={:0.5f}".format(eval_results_val[0], eval_results_val[1]) )

    #--------------------
    # モデルの保存
    #--------------------
    if( args.train_mode in ["train"] ):
        save_checkpoint( model.model, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final') )

    #--------------------
    # モデルの推論処理
    #--------------------
    y_pred_train = model.predict(X_valid)
    y_pred_test = model.predict(X_test)
    if( args.debug ):
        print( "y_pred_train.shape", y_pred_train.shape )
        print( "y_pred_test.shape", y_pred_test.shape )

    #================================
    # 可視化処理
    #================================
    # 生成画像
    n_images = 10
    for i, name in enumerate(image_names_valid[0:n_images-1]):
        cv2.imwrite( os.path.join( args.results_dir, args.exper_name, "valid", "images", name ), X_train_img.squeeze()[i,:,:] * 255 )
        cv2.imwrite( os.path.join( args.results_dir, args.exper_name, "valid", "masks", name ), y_pred_train.squeeze()[i,:,:] * 255 )

    for i, name in enumerate(test_image_names[0:n_images-1]):
        cv2.imwrite( os.path.join( args.results_dir, args.exper_name, "test", "images", name ), X_test_img.squeeze()[i,:,:] * 255 )
        cv2.imwrite( os.path.join( args.results_dir, args.exper_name, "test", "masks", name ), y_pred_test.squeeze()[i,:,:] * 255 )

    # 損失関数
    if( args.train_mode in ["train"] ):
        model.plot_loss( os.path.join(args.results_dir, args.exper_name, "losees.png" ) )

    # IoU
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

    # 元画像と生成マスク画像の重ね合わせ（test）
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

    # 元画像と生成マスク画像の重ね合わせ（valid）
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

    #================================
    # Kaggle API での submit
    #================================
    # RLE [Run Length Encoding] 形式で提出のため生成画像を元の画像サイズに変換
    y_pred_test_org = np.zeros( (len(y_pred_test), args.image_height_org, args.image_width_org), dtype=np.float32 )
    for i in range(len(y_pred_test)):
        #y_pred_test_org[i] = cv2.resize( y_pred_test[i].squeeze(), (args.image_height_org, args.image_width_org), interpolation = cv2.INTER_NEAREST )
        y_pred_test_org[i] = resize( y_pred_test[i].squeeze(), (args.image_height_org, args.image_width_org), mode='constant', preserve_range=True )

    # 提出用データに値を設定
    y_sub = { name.split(".png")[0] : convert_rle(np.round(y_pred_test_org[i] > threshold_best)) for i,name in enumerate(test_image_names) }
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
