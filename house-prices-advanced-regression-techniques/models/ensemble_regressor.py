# -*- coding:utf-8 -*-
import os
import numpy as np
from tqdm import tqdm

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import _name_estimators
from sklearn.base import clone

class EnsembleRegressor( BaseEstimator, RegressorMixin ):
    """
    アンサンブルモデルの回帰器 regressor の自作クラス.
    scikit-learn ライブラリの推定器 estimator の基本クラス BaseEstimator, RegressorMixin を継承している.
    """
    def __init__( self, regressors, weights = None, fitting = None, debug = False ):
        """
        Args :
            regressors : list <regressors オブジェクト>
                回帰器のオブジェクトのリスト
            weights : list <float>
                各回帰器の対する重みの値のリスト : __init()__ の引数と同名のオブジェクトの属性
            fitting : list<bool>
                各回帰器の対する学習を行うかのフラグのリスト
        """
        self.regressors = regressors
        self.fitting = fitting
        self.fitted_regressors = regressors
        self.weights = weights
        self.debug = debug
        if regressors != None:
            self.n_classifier = len( regressors )
        else:
            self.n_classifier = 0

        self.encoder = LabelEncoder()

        # regressors　で指定した各オブジェクトの名前
        if regressors != None:
            self.named_regressors = { key: value for key, value in _name_estimators(regressors) }
        else:
            self.named_regressors = {}

        if( self.debug ):
            for i, named_classifier in enumerate(self.named_regressors):
                print( "name {} : {}".format(i, self.named_regressors[named_classifier]) )

        if fitting == None:
            fitting = []
            for i in range(len(self.regressors)):
                fitting.append(True)

        return

    def fit( self, X_train, y_train ):
        """
        識別器に対し, 指定されたデータで fitting を行う関数
        scikit-learn ライブラリの識別器 : classifiler, 推定器 : estimator が持つ共通関数

        [Input]
            X_train : np.ndarray ( shape = [n_samples, n_features] )
                トレーニングデータ（特徴行列）

            y_train : np.ndarray ( shape = [n_samples] )
                トレーニングデータ用のクラスラベル（教師データ）のリスト

        [Output]
            self : 自身のオブジェクト

        """
        # self.regressors に設定されている分類器のクローン clone(reg) で fitting
        self.fitted_regressors = []
        for i, reg in enumerate( tqdm(self.regressors, desc = "fitting regressors") ):
            if( self.fitting[i] ):
                # clone() : 同じパラメータの 推定器を deep copy
                fitted_reg = clone(reg).fit( X_train, y_train )
            else:
                fitted_reg = clone(reg)
                #fitted_reg = reg

            self.fitted_regressors.append( fitted_reg )

        return self # scikit-learn の fit() は self を返す

    def predict( self, X_test ):
        """
        識別器に対し, fitting された結果を元に, クラスラベルの予想値を返す関数

        [Input]
            X_test : np.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列
        [Output]
            vote_results : np.ndaary ( shape = [n_samples] )
                予想結果（クラスラベル）
        """
        # 各弱回帰器 reg の predict_prpba() 結果を predictions (list) に格納
        predict_probas = np.asarray( [ reg.predict(X_test) for reg in self.fitted_regressors ] )     # shape = [n_classifer, n_features]

        # 平均化
        ave_probas = np.average( predict_probas, axis = 0, weights = self.weights )

        return ave_probas



