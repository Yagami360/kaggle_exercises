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
from kaggle.api.kaggle_api_extended import KaggleApi

# keras
import keras
from keras.preprocessing import image
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
import keras.backend.tensorflow_backend as KTF

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# 機械学習モデル
import catboost

# 自作モジュール
from dataset import load_dataset
from models import SklearnImageClassifier, CatBoostImageClassifier
from models import KerasMLPImageClassifier, KerasResNet50ImageClassifier, KerasMNISTResNetImageClassifier
from utils import save_checkpoint, load_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="single_model", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="imaterialist-fashion-2019-FGVC6")
    parser.add_argument("--train_mode", choices=["train", "test", "eval"], default="train", help="")
    parser.add_argument("--classifier", choices=["catboost", "mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"], default="catboost", help="分類器モデルの種類")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_paths', action='append', help="モデルの読み込みファイルのパス")
    parser.add_argument("--n_epoches", type=int, default=10, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=0.001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_height', type=int, default=512, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=512, help="入力画像の幅（pixel単位）")
    parser.add_argument("--n_channels", type=int, default=3, help="チャンネル数")    
    parser.add_argument("--n_classes", type=int, default=46, help="ラベル数")    
    parser.add_argument("--n_samplings", type=int, default=100, help="ラベル数")    

    parser.add_argument('--data_augument', action='store_false', help="Data Augumentation を行うか否か")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="cpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 実験名を自動的に変更
    if( args.exper_name == "single_model" ):
        if( args.train_mode in ["test", "eval"] ):
            args.exper_name = "test_" + args.exper_name
        args.exper_name += "_" + args.classifier
        if( args.classifier in ["mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"] ):
            args.exper_name += "_ep" + str(args.n_epoches)
            args.exper_name += "_b" + str(args.batch_size)
        if( args.classifier in ["catboost", "mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"] ):
            args.exper_name += "_lr{}".format(args.lr)
        args.exper_name += "_k" + str(args.n_splits)
        if( args.data_augument ):
            args.exper_name += "_da"

    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if( args.train_mode in ["train"] ):
        if( args.classifier in ["catboost", "mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"] ):
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
    X_train, y_train, X_test = load_dataset( dataset_dir = args.dataset_dir, image_height = args.image_height, image_width = args.image_width, n_channels = args.n_channels, n_classes = args.n_classes, one_hot_encode = False, n_samplings = args.n_samplings )
    y_pred_train = np.zeros((len(y_train),))
    if( args.debug ):
        print( "X_train.shape : ", X_train.shape )
        print( "y_train.shape : ", y_train.shape )
        print( "X_test.shape : ", X_test.shape )
        print( "y_pred_train.shape : ", y_pred_train.shape )

    #================================
    # 前処理
    #================================
    # データオーギュメントとバッチ単位で学習のための DataGenerator
    if( args.data_augument ):
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
            horizontal_flip = False,                  # randomly flip images
            vertical_flip = False,                    # randomly flip images
        )
        datagen.fit(X_train)
    else:
        datagen = ImageDataGenerator()
        datagen.fit(X_train)

    #================================
    # モデルの学習 & 推論
    #================================    
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_preds_test = []
    k = 0
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        # seed 値の固定
        np.random.seed(args.seed+k)
        random.seed(args.seed+k)
        
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
        y_train_fold, y_valid_fold = y_train[train_index], y_train[valid_index]
        if( args.debug ):
            print( "X_train_fold.shape : ", X_train_fold.shape )
            print( "X_valid_fold.shape : ", X_valid_fold.shape )
            print( "y_train_fold.shape : ", y_train_fold.shape )
            print( "y_valid_fold.shape : ", y_valid_fold.shape )

        #--------------------
        # keras の call back
        #--------------------
        if( args.classifier in ["mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"] ):
            """
            # 各エポック終了毎のモデルのチェックポイント保存用 call back
            callback_checkpoint = ModelCheckpoint( 
                filepath = os.path.join(args.save_checkpoints_dir, args.exper_name, "ep{epoch:03d}.hdf5" ), 
                monitor = 'loss', 
                verbose = 1, 
                save_weights_only = True,   # 
                save_best_only = False,     # 精度がよくなった時だけ保存するかどうか指定。False の場合は毎 epoch 毎保存．
                mode = 'auto',              # 
                period = 1                  # 何エポックごとに保存するか
            )
            callbacks = [ callback_checkpoint ]
            """
            callbacks = None

        #--------------------
        # モデルの定義
        #--------------------
        if( args.classifier == "catboost" ):
            if( args.device == "gpu" ): 
                model = CatBoostImageClassifier( model = catboost.CatBoostClassifier( loss_function="MultiClass", iterations = 1000, learning_rate = args.lr, task_type="GPU", devices='0:1' ), use_valid = True, debug = args.debug )  # iterations = (trees / (epochs * batches)
            else:
                model = CatBoostImageClassifier( model = catboost.CatBoostClassifier( loss_function="MultiClass", iterations = 1000, learning_rate = args.lr ), use_valid = True, debug = args.debug )
        elif( args.classifier == "mlp" ):
            model = KerasMLPImageClassifier( 
                n_input_dim = X_train_fold.shape[1] * X_train_fold.shape[2] * X_train_fold.shape[3], n_classes = args.n_classes, 
                n_epoches = args.n_epoches, batch_size = args.batch_size, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2,
                use_valid = True, one_hot_encode = True, callbacks = callbacks, use_datagen = False, datagen = datagen, debug = args.debug
            )
        elif( args.classifier == "resnet50" ):
            model = KerasResNet50ImageClassifier( 
                image_height = args.image_height, image_width = args.image_width, n_channles = 3, n_classes = args.n_classes, 
                n_epoches = args.n_epoches, batch_size = args.batch_size, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2,
                pretrained = False, train_only_fc = False,
                use_valid = True, one_hot_encode = True, callbacks = callbacks, use_datagen = True, datagen = datagen, debug = args.debug
            )
        elif( args.classifier == "pretrained_resnet50" ):
            model = KerasResNet50ImageClassifier( 
                image_height = args.image_height, image_width = args.image_width, n_channles = 3, n_classes = args.n_classes, 
                n_epoches = args.n_epoches, batch_size = args.batch_size, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2,
                pretrained = True, train_only_fc = False,
                use_valid = True, one_hot_encode = True, callbacks = callbacks, use_datagen = True, datagen = datagen, debug = args.debug
            )
        elif( args.classifier == "pretrained_resnet50_fc" ):
            model = KerasResNet50ImageClassifier( 
                image_height = args.image_height, image_width = args.image_width, n_channles = 3, n_classes = args.n_classes, 
                n_epoches = args.n_epoches, batch_size = args.batch_size, lr = args.lr, beta1 = args.beta1, beta2 = args.beta2,
                pretrained = True, train_only_fc = True,
                use_valid = True, one_hot_encode = True, callbacks = callbacks, use_datagen = True, datagen = datagen, debug = args.debug
            )

        # モデルを読み込む
        if( args.classifier in ["mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"] ):
            if( args.load_checkpoints_paths != None ):
                if not ( args.load_checkpoints_paths[k] == '' and os.path.exists(args.load_checkpoints_paths[k]) ):
                    load_checkpoint(model.model, args.load_checkpoints_paths[k] )
        elif( args.classifier in ["catboost"] ):
            if( args.load_checkpoints_paths != None ):
                if not ( args.load_checkpoints_paths[k] == '' and os.path.exists(args.load_checkpoints_paths[k]) ):
                    model.model.load_model( args.load_checkpoints_paths[k], format = "json" )

        #--------------------
        # モデルの学習処理
        #--------------------
        if( args.train_mode in ["train"] ):
            model.fit(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold)
        elif( args.train_mode in ["eval"] ):
            if( args.classifier in ["mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"] ):
                eval_results_train = model.evaluate( X_train_fold, y_train_fold )
                eval_results_val = model.evaluate( X_valid_fold, y_valid_fold )
                print( "loss [train]={:0.5f}, accuracy [train]={:0.5f}".format(eval_results_train[0], eval_results_train[1]) )
                print( "loss [valid]={:0.5f}, accuracy [valid]={:0.5f}".format(eval_results_val[0], eval_results_val[1]) )

        #--------------------
        # モデルの推論処理
        #--------------------
        y_pred_train[valid_index] = model.predict(X_valid_fold)
        y_pred_test = model.predict(X_test)
        y_preds_test.append(y_pred_test)
    
        #--------------------
        # 可視化処理
        #--------------------
        # 損失関数
        if( args.train_mode in ["train"] ):
            model.plot_loss( os.path.join(args.results_dir, args.exper_name, "losees_k{}.png".format(k) ) )

        #--------------------
        # モデルの保存
        #--------------------
        k += 1
        if( args.train_mode in ["train"] ):
            if( args.classifier in ["mlp", "resnet50", "pretrained_resnet50", "pretrained_resnet50_fc"] ):
                save_checkpoint( model.model, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_k{}_final'.format(k)) )
            elif( args.classifier in ["catboost"] ):
                model.model.save_model( os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_k{}_final.json'.format(k)), format = "json" )

    # k-fold CV で平均化
    y_preds_test = sum(y_preds_test) / len(y_preds_test)

    # accuracy
    accuracy = (y_train == y_pred_train).sum()/len(y_pred_train)
    print( "accuracy [k-fold CV train-valid] : {:0.5f}".format(accuracy) )

    #================================
    # 可視化処理
    #================================
    # 判定結果を画像表示
    labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    fig, axes = plt.subplots(5, 5, figsize=(16, 20), facecolor='w')
    for i, ax in enumerate(axes.ravel()):            
        ax.set_title( 'correct : {}, pred : {}'.format( labels[y_train[i]], labels[y_pred_train[i]]) )
        if( args.n_channels == 1 ):
            img = Image.fromarray( np.uint8(X_train[i,:,:,0]*255), mode = "L" )
        else:
            img = Image.fromarray( np.uint8(X_train[i,:,:]*255), mode = "RGB" )

        ax.imshow(img)

    fig.savefig( os.path.join(args.results_dir, args.exper_name, "classification.png"), dpi = 100, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = list(map(int, y_preds_test))
    df_submission['Label'] = y_sub
    df_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
