# -*- coding:utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.pipeline import _name_estimators
from sklearn.base import clone


def predict_from_submit_files( key, weights = [], submit_files = [] ):
    predict_probas = []
    for i, submit_file in enumerate( submit_files ):
        ds_submission = pd.read_csv( submit_file )
        predict_proba = ds_submission[key]
        predict_probas.append( predict_proba )

    ave_probas = np.average( predict_probas, axis = 0, weights = weights )
    return ave_probas


class WeightAverageEnsembleRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, regressors, weights = None, fitting = None, clone = False, debug = False ):
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
        self.fitted_regressors = regressors
        self.weights = weights
        self.clone = clone
        self.debug = debug
        if regressors != None:
            self.n_classifier = len( regressors )
        else:
            self.n_classifier = 0

        # regressors　で指定した各オブジェクトの名前
        if regressors != None:
            self.named_regressors = { key: value for key, value in _name_estimators(regressors) }
        else:
            self.named_regressors = {}

        if( self.debug ):
            for i, named_classifier in enumerate(self.named_regressors):
                print( "name {} : {}".format(i, self.named_regressors[named_classifier]) )

        return

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        # self.regressors に設定されている分類器のクローン clone(reg) で fitting
        self.fitted_regressors = []
        for i, reg in enumerate( tqdm(self.regressors, desc = "fitting regressors") ):
            # clone() : 同じパラメータの 推定器を deep copy
            if( self.clone ):
                fitted_reg = clone(reg).fit( X_train, y_train, X_valid, y_valid )
            else:
                fitted_reg = reg.fit( X_train, y_train, X_valid, y_valid )

            self.fitted_regressors.append( fitted_reg )

        return self # scikit-learn の fit() は self を返す

    def predict( self, X_test ):
        # 各弱回帰器 reg の predict_prpba() 結果を predictions (list) に格納
        predict_probas = np.asarray( [ reg.predict(X_test) for reg in self.fitted_regressors ] )     # shape = [n_classifer, n_features]

        # 平均化
        ave_probas = np.average( predict_probas, axis = 0, weights = self.weights )

        return ave_probas


class StackingEnsembleRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, regressors, final_regressors, second_regressors = None, n_splits = 4, clone = False, seed = 72 ):
        self.regressors = regressors
        self.fitted_regressors = regressors
        self.final_regressors = final_regressors
        self.second_regressors = second_regressors
        self.fitted_second_regressors = second_regressors

        self.n_classifier = len( regressors )
        if( second_regressors != None ):
            self.n_second_regressors = len( second_regressors )
        else:
            self.n_second_regressors = 0

        self.n_splits = n_splits
        self.clone = clone
        self.seed = seed
        self.accuracy = None

        # classifiers　で指定した各オブジェクトの名前
        if regressors != None:
            self.named_regressors = { key: value for key, value in _name_estimators(regressors) }
        else:
            self.named_regressors = {}

        for i, named_regressor in enumerate(self.named_regressors):
            print( "name {} : {}".format(i, self.named_regressors[named_regressor]) )

        return

    def fit( self, X_train, y_train, X_test ):
        #--------------------------------
        # １段目の k-fold CV での学習 & 推論
        #--------------------------------
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_preds_train = np.zeros( (self.n_classifier, len(y_train)) )
        y_preds_test = np.zeros( (self.n_classifier, self.n_splits, len(X_test)) )

        k = 0
        for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
            #--------------------
            # データセットの分割
            #--------------------
            X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

            #-------------------
            # 各モデルの学習処理
            #-------------------
            self.fitted_regressors = []
            for i, reg in enumerate( tqdm(self.regressors, desc="fitting regressors") ):
                if( self.clone ):
                    fitted_reg = clone(reg).fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )
                else:
                    fitted_reg = reg.fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

                self.fitted_regressors.append( fitted_reg )

            #-------------------
            # 各モデルの推論処理
            #-------------------
            for i, reg in enumerate(self.fitted_regressors):
                y_preds_train[i][valid_index] = reg.predict(X_valid_fold)
                y_preds_test[i][k] = reg.predict(X_test)

            k += 1

        # テストデータに対する予測値の k-fold CV の平均をとる
        y_preds_test = np.mean( y_preds_test, axis=1 )
        #print( "y_preds_test.shape : ", y_preds_test.shape )

        # 各モデルの予想値をスタッキング
        y_preds_train_stack = y_preds_train[0]
        y_preds_test_stack = y_preds_test[0]
        for i in range(self.n_classifier - 1):
            y_preds_train_stack = np.column_stack( (y_preds_train_stack, y_preds_train[i+1]) )
            y_preds_test_stack = np.column_stack( (y_preds_test_stack, y_preds_test[i+1]) )

        y_preds_train = y_preds_train_stack
        y_preds_test = y_preds_test_stack
        #print( "y_preds_train.shape : ", y_preds_train.shape )
        #print( "y_preds_test.shape : ", y_preds_test.shape )

        # 予測値を新たな特徴量としてデータフレーム作成
        X_train = pd.DataFrame(y_preds_train)
        X_test = pd.DataFrame(y_preds_test)

        #--------------------------------
        # ２段目の k-fold CV での学習 & 推論
        #--------------------------------
        if( self.second_regressors != None ):
            kf = StratifiedKFold( n_splits=self.n_splits, shuffle=True, random_state=self.seed )
            y_preds_train = np.zeros( (self.n_second_regressors, len(y_train)) )
            y_preds_test = np.zeros( (self.n_second_regressors, self.n_splits, len(X_test)) )
            
            k = 0
            for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
                #--------------------
                # データセットの分割
                #--------------------
                X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
                y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

                #-------------------
                # 各モデルの学習処理
                #-------------------
                self.fitted_second_regressors = []
                for i, reg in enumerate( tqdm(self.second_regressors, desc="fitting second regressors") ):
                    if( self.clone ):
                        fitted_reg = clone(reg).fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )
                    else:
                        fitted_reg = reg.fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

                    self.fitted_second_regressors.append( fitted_reg )

                #-------------------
                # 各モデルの推論処理
                #-------------------
                for i, reg in enumerate(self.fitted_second_regressors):
                    y_preds_train[i][valid_index] = reg.predict(X_valid_fold)
                    y_preds_test[i][k] = reg.predict(X_test)

                k += 1

            # テストデータに対する予測値の k-fold CV の平均をとる
            y_preds_test = np.mean( y_preds_test, axis=1 )
            #print( "y_preds_test.shape : ", y_preds_test.shape )

            # 各モデルの予想値をスタッキング
            y_preds_train_stack = y_preds_train[0]
            y_preds_test_stack = y_preds_test[0]
            for i in range(self.n_second_regressors - 1):
                y_preds_train_stack = np.column_stack( (y_preds_train_stack, y_preds_train[i+1]) )
                y_preds_test_stack = np.column_stack( (y_preds_test_stack, y_preds_test[i+1]) )

            y_preds_train = y_preds_train_stack
            y_preds_test = y_preds_test_stack

            # 予測値を新たな特徴量としてデータフレーム作成
            X_train = pd.DataFrame(y_preds_train)
            X_test = pd.DataFrame(y_preds_test)

        #--------------------------------
        # 最終層の k-fold CV での学習 & 推論
        #--------------------------------
        y_preds_train = np.zeros( (len(y_train)) )
        y_preds_test = np.zeros( (self.n_splits, len(X_test)) )
        k = 0
        for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
            #--------------------
            # データセットの分割
            #--------------------
            X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

            #-------------------
            # 各モデルの学習処理
            #-------------------
            if( self.clone ):
                clone(self.final_regressors).fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )
            else:
                self.final_regressors.fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

            #-------------------
            # 各モデルの推論処理
            #-------------------
            y_preds_train[valid_index] = self.final_regressors.predict(X_valid_fold)
            y_preds_test[k] = self.final_regressors.predict(X_test)
            k += 1

        # テストデータに対する予測値の平均をとる
        self.y_preds_test = np.mean( y_preds_test, axis=0 )
        self.y_preds_train = y_preds_train
        return self

    """
    def predict( self, X_test ):
        return self.y_preds_test
    """

    """
    def predict_proba( self, X_test ):
        return self.y_preds_test
    """