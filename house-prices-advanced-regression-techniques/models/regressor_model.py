import os
import numpy as np
#from apex import amp

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import RegressorMixin            # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.utils.estimator_checks import check_estimator

from sklearn.pipeline import _name_estimators       # 
from sklearn.base import clone                      #

class RegressorXGBoost( BaseEstimator, RegressorMixin ):
    def __init__( self, model, debug = False ):
        self.model = model
        self.debug = debug
        return

    def fit( self, X_train, y_train ):
        #X_train = X_train.reshape(X_train.shape[0],-1)  # shape = [N,H,W,C] -> [N, H*W*C] 
        #X_train = X_train.detach().cpu().numpy().reshape(X_train.shape[0],-1)   # shape = [N,C,H,W] -> [N, C*H*W] 
        self.model.fit(X_train, y_train, verbose = self.debug )
        return self

    def predict(self, X_test):
        #X_test = X_test.detach().cpu().numpy().reshape(X_test.shape[0],-1)
        predicts = self.model.predict(X_test)
        predicts = np.argmax(predicts, axis = 1)
        """
        if( self.debug ):
            print( "[XBboost] predicts.shape : ", predicts.shape )
            #print( "[XBboost] predicts[0:10] : ", predicts[0:5] )
        """
        return predicts

    def predict_proba(self, X_test):
        #X_test = X_test.detach().cpu().numpy().reshape(X_test.shape[0],-1)
        predicts = self.model.predict_proba(X_test)
        #predicts = predicts[:,1] 
        """
        if( self.debug ):
            print( "[XBboost] predicts.shape : ", predicts.shape )
            #print( "[XBboost] predicts[0:10] : ", predicts[0:5] )
        """
        return predicts