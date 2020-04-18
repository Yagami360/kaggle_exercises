# -*- coding:utf-8 -*-
import os
import numpy as np

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import ClassifierMixin            # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.utils.estimator_checks import check_estimator

from sklearn.pipeline import _name_estimators       # 
from sklearn.base import clone                      #

# keras
from keras.utils import to_categorical
#from utils import save_checkpoint, load_checkpoint

class ImageClassifierKeras( BaseEstimator, ClassifierMixin ):
    def __init__( self, model, n_classes = 2, debug = False ):
        self.model = model
        self.n_classes = 2
        self.debug = debug
        return

    """
    def load_check_points( self, load_checkpoints_path ):
        if not load_checkpoints_path == '' and os.path.exists(load_checkpoints_path):
            load_checkpoint(self.model, load_checkpoints_path )
        return
    """

    """
    def compile( self, loss, optimizer, metrics ):
        self.model.compile( loss, optimizer, metrics )
        return
    """

    def fit(self, X_train, y_train, one_hot_encode = False ):
        if( one_hot_encode == False ):
            y_train = to_categorical(y_train)

        self.model.fit( X_train, y_train, shuffle = True, verbose = 1 )
        return self

    def predict(self, X_test):
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 ) 
        predicts = np.argmax(predicts, axis = 1)
        if( self.debug ):
            print( "[keras] predicts.shape : ", predicts.shape )
            print( "[keras] predicts[0:5] : ", predicts[0:5] )

        return predicts

    def predict_proba(self, X_test):
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 ) 
        #predicts = predicts[:,1] 
        if( self.debug ):
            print( "[keras] predicts.shape : ", predicts.shape )
            print( "[keras] predicts[0:5] : ", predicts[0:5] )

        return predicts


class ImageClassifierSklearn( BaseEstimator, ClassifierMixin ):
    def __init__( self, model, debug = False ):
        self.model = model
        self.debug = debug
        return

    def fit( self, X_train, y_train ):
        X_train = X_train.reshape(X_train.shape[0],-1)  # shape = [N,H,W,C] -> [N, H*W*C] 
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict(X_test)
        predicts = np.argmax(predicts, axis = 1)
        if( self.debug ):
            print( "[sklearn] predicts.shape : ", predicts.shape )
            print( "[sklearn] predicts[0:5] : ", predicts[0:5] )

        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict_proba(X_test)
        #predicts = predicts[:,1] 
        if( self.debug ):
            print( "[sklearn] predicts.shape : ", predicts.shape )
            print( "[sklearn] predicts[0:5] : ", predicts[0:5] )

        return predicts


class ImageClassifierXGBoost( BaseEstimator, ClassifierMixin ):
    def __init__( self, model, debug = False ):
        self.model = model
        self.debug = debug
        return

    def fit( self, X_train, y_train ):
        X_train = X_train.reshape(X_train.shape[0],-1)  # shape = [N,H,W,C] -> [N, H*W*C] 
        self.model.fit(X_train, y_train, verbose = self.debug )
        return self

    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict(X_test)
        predicts = np.argmax(predicts, axis = 1)
        if( self.debug ):
            print( "[XBboost] predicts[0:10] : ", predicts[0:5] )

        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict_proba(X_test)
        #predicts = predicts[:,1] 
        if( self.debug ):
            print( "[XBboost] predicts.shape : ", predicts.shape )
            print( "[XBboost] predicts[0:10] : ", predicts[0:5] )

        return predicts
