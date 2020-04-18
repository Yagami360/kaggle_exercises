import os
import numpy as np
#from apex import amp

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import ClassifierMixin            # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.utils.estimator_checks import check_estimator

from sklearn.pipeline import _name_estimators       # 
from sklearn.base import clone                      #

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# 自作クラス
from utils import save_checkpoint, load_checkpoint


class ImageClassifierPyTorch( nn.Module, BaseEstimator, ClassifierMixin ):
    def __init__(self, device, model, debug = False ):
        super( ImageClassifierPyTorch, self ).__init__()
        self.device = device
        self.model = model
        self.debug = debug

        # optimizer の設定
        pass

        # loss 関数の設定
        pass

        return

    def load_check_point( self, load_checkpoints_path ):
        if not load_checkpoints_path == '' and os.path.exists(load_checkpoints_path):
            load_checkpoint(self.model, self.device, load_checkpoints_path )
            print( "load check points" )
        return

    def set_optimizer(self):
        # 未実装
        return

    def set_loss_fn(self):
        # 未実装
        return

    def fit( self, X_train, y_train ):
        """
        [args]
            X_train : <tensor> 学習用画像
            X_train : <tensor> ラベル値
        """
        self.model.train()

        # 未実装
        pass

        return self

    def predict( self, X_test ):
        self.model.eval()

        # データをモデルに流し込む
        with torch.no_grad():
            output = self.model( X_test )

        # 確率値が最大のラベル 0~9 を予想ラベルとする。
        _, predicts = torch.max( output.data, dim = 1 )
        predicts = predicts.detach().cpu().numpy()
        """
        if( self.debug ):
            print( "[PyTorch] predicts.shape : ", predicts.shape )
        """
        
        return predicts


    def predict_proba( self, X_test ):
        """
        [args]
            X_test : <tensor> テスト用画像 / shape=(N,C,H,W)
        """
        self.model.eval()

        # データをモデルに流し込む
        with torch.no_grad():
            output = self.model( X_test )

        # 確率値で出力
        #predicts = F.softmax(output, dim=1)[:, 1]
        predicts = F.softmax(output, dim=1)
        predicts = predicts.detach().cpu().numpy()
        """
        if( self.debug ):
            print( "[PyTorch] predicts.shape : ", predicts.shape )
        """
        return predicts


class ImageClassifierSklearn( BaseEstimator, ClassifierMixin ):
    def __init__( self, model, debug = False ):
        self.model = model
        self.debug = debug
        return

    def fit( self, X_train, y_train ):
        X_train = X_train.reshape(X_train.shape[0],-1)  # shape = [N,H,W,C] -> [N, H*W*C] 
        #X_train = X_train.detach().cpu().numpy().reshape(X_train.shape[0],-1)   # shape = [N,C,H,W] -> [N, C*H*W] 
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        X_test = X_test.detach().cpu().numpy().reshape(X_test.shape[0],-1)   # shape = [N,C,H,W] -> [N, C*H*W] 
        predicts = self.model.predict(X_test)
        predicts = np.argmax(predicts, axis = 1)
        """
        if( self.debug ):
            print( "[sklearn] predicts.shape : ", predicts.shape )
            #print( "[sklearn] predicts[0:5] : ", predicts[0:5] )
        """
        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.detach().cpu().numpy().reshape(X_test.shape[0],-1)   # shape = [N,C,H,W] -> [N, C*H*W] 
        predicts = self.model.predict_proba(X_test)
        #predicts = predicts[:,1] 
        """
        if( self.debug ):
            print( "[sklearn] predicts.shape : ", predicts.shape )
            #print( "[sklearn] predicts[0:5] : ", predicts[0:5] )
        """
        return predicts


class ImageClassifierXGBoost( BaseEstimator, ClassifierMixin ):
    def __init__( self, model, debug = False ):
        self.model = model
        self.debug = debug
        return

    def fit( self, X_train, y_train ):
        X_train = X_train.reshape(X_train.shape[0],-1)  # shape = [N,H,W,C] -> [N, H*W*C] 
        #X_train = X_train.detach().cpu().numpy().reshape(X_train.shape[0],-1)   # shape = [N,C,H,W] -> [N, C*H*W] 
        self.model.fit(X_train, y_train, verbose = self.debug )
        return self

    def predict(self, X_test):
        X_test = X_test.detach().cpu().numpy().reshape(X_test.shape[0],-1)
        predicts = self.model.predict(X_test)
        predicts = np.argmax(predicts, axis = 1)
        """
        if( self.debug ):
            print( "[XBboost] predicts.shape : ", predicts.shape )
            #print( "[XBboost] predicts[0:10] : ", predicts[0:5] )
        """
        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.detach().cpu().numpy().reshape(X_test.shape[0],-1)
        predicts = self.model.predict_proba(X_test)
        #predicts = predicts[:,1] 
        """
        if( self.debug ):
            print( "[XBboost] predicts.shape : ", predicts.shape )
            #print( "[XBboost] predicts[0:10] : ", predicts[0:5] )
        """
        return predicts
