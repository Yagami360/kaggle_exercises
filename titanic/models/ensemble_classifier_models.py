# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator              # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import ClassifierMixin            # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


class WeightAverageEnsembleClassifier( BaseEstimator, ClassifierMixin ):
    """
    アンサンブルモデルの識別器 classifier の自作クラス.
    scikit-learn ライブラリの推定器 estimator の基本クラス BaseEstimator, ClassifierMixin を継承している.
    """
    def __init__( self, classifiers, weights = None, vote_method = "majority_vote", clone = False ):
        """
        Args :
            classifiers : list <classifier オブジェクト>
                分類器のクラスのオブジェクトのリスト
            weights : list <float>
                各分類器の対する重みの値のリスト : __init()__ の引数と同名のオブジェクトの属性
            vote_method : str ( "majority_vote" or "probability_vote" )
                アンサンブルによる最終的な判断判断手法 : __init()__ の引数と同名のオブジェクトの属性
                "majority_vote"    : 弱識別器の多数決で決定する.多数決方式 (＝クラスラベルの argmax() 結果）
                "probability_vote" : 弱識別器の重み付け結果で決定する.（＝クラスの所属確率の argmax() 結果）
        """
        self.classifiers = classifiers
        self.fitted_classifiers = classifiers
        self.weights = weights
        self.n_classes = 0
        self.n_classifier = len( classifiers )
        self.vote_method = vote_method
        self.clone = clone
        self.encoder = LabelEncoder()

        # classifiers　で指定した各オブジェクトの名前
        if classifiers != None:
            self.named_classifiers = { key: value for key, value in _name_estimators(classifiers) }
        else:
            self.named_classifiers = {}

        for i, named_classifier in enumerate(self.named_classifiers):
            print( "name {} : {}".format(i, self.named_classifiers[named_classifier]) )

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
        # LabelEncoder クラスを使用して, クラスラベルが 0 から始まるようにする.
        # これは, self.predict() 関数内の np.argmax() 関数呼び出し時に重要となるためである.
        self.encoder.fit( y_train )
        self.n_classes = self.encoder.classes_

        # self.classifiers に設定されている分類器のクローン clone(clf) で fitting
        self.fitted_classifiers = []
        for clf in self.classifiers:
            if( self.clone ):
                # clone() : 同じパラメータの 推定器を生成
                fitted_clf = clone(clf).fit( X_train, self.encoder.transform(y_train) )
            else:
                fitted_clf = clf.fit( X_train, self.encoder.transform(y_train) )

            self.fitted_classifiers.append( fitted_clf )

        return self

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
        #------------------------------------------------------------------------------------------------------
        # アンサンブルの最終決定方式 vote_method が, 各弱識別器の重み付け方式 "probability_vote" のケース
        #------------------------------------------------------------------------------------------------------
        if self.vote_method == "probability_vote":
            # np.argmax() : 指定した配列の最大要素のインデックスを返す
            # axis : 最大値を読み取る軸の方向 ( axis = 1 : shape が２次元配列 行方向)
            vote_results = np.argmax( self.predict_proba(X_test), axis = 1 )

        #------------------------------------------------------------------------------------------------------        
        # アンサンブルの最終決定方式 vote_method が, 多数決方式 "majority_vote" のケース
        #------------------------------------------------------------------------------------------------------
        else:
            # 各弱識別器 clf の predict() 結果を predictions (list) に格納
            predictions = [ clf.predict(X_test) for clf in self.fitted_classifiers ]
            #print( "predictions : \n", predictions)
            #print( "predictions.shape : \n", predictions[0].shape)

            # predictions を 転置し, 行と列 (shape) を反転
            # np.asarray() :  np.array とほとんど同じように使えるが, 引数に ndarray を渡した場合は配列のコピーをせずに引数をそのまま返す。
            predictions = np.asarray( predictions ).T
            #print( "predictions : \n", predictions)
            #print( "predictions.shape : \n", predictions[0].shape)

            # 各サンプルの所属クラス確率に重み付けで足し合わせた結果が最大となるようにし、列番号を返すようにする.
            # この処理を np.apply_along_axis() 関数を使用して実装
            # np.apply_along_axis() : Apply a function to 1-D slices along the given axis.
            # Execute func1d(a, *args) where func1d operates on 1-D arrays and a is a 1-D slice of arr along axis.
            vote_results = np.apply_along_axis(
                               lambda x : np.argmax( np.bincount( x, weights = self.weights ) ),    # 
                               axis = 1,                                                            #
                               arr = predictions                                                    # ndarray : Input array
                           )

        # vote_results を LabelEncoder で逆行列化して, shape を反転
        vote_results = self.encoder.inverse_transform( vote_results )
        return vote_results


    def predict_proba( self, X_test ):
        """
        識別器に対し, fitting された結果を元に, クラスの所属確率の予想値を返す関数

        [Input]
            X_test : np.ndarry ( shape = [n_samples, n_features] )
                予想したい特徴行列

        [Output]
            ave_probas : np.nadarry ( shape = [n_samples, n_classes] )
                各サンプルの所属クラス確率に重み付けした結果の平均確率
        """
        # 各弱識別器 clf の predict_prpba() 結果を predictions (list) に格納
        #predict_probas = [ clf.predict_proba(X_test) for clf in self.fitted_classifiers ]
        #print( "EnsembleLearningClassifier.predict_proba() { predict_probas } : \n", predict_probas )
        predict_probas = np.asarray( [ clf.predict_proba(X_test) for clf in self.fitted_classifiers ] )

        # 平均化
        ave_probas = np.average( predict_probas, axis = 0, weights = self.weights )
        #print( "EnsembleLearningClassifier.predict_proba() { ave_probas } : \n", ave_probas )

        return ave_probas


class StackingEnsembleClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( self, classifiers, final_classifiers, second_classifiers = None, n_splits = 4, clone = False, seed = 72 ):
        self.classifiers = classifiers
        self.fitted_classifiers = classifiers
        self.final_classifiers = final_classifiers
        self.second_classifiers = second_classifiers
        self.fitted_second_classifiers = second_classifiers

        self.n_classifier = len( classifiers )
        if( second_classifiers != None ):
            self.n_second_classifier = len( second_classifiers )
        else:
            self.n_second_classifier = 0

        self.n_splits = n_splits
        self.clone = clone
        self.seed = seed
        self.accuracy = None

        # classifiers　で指定した各オブジェクトの名前
        if classifiers != None:
            self.named_classifiers = { key: value for key, value in _name_estimators(classifiers) }
        else:
            self.named_classifiers = {}

        for i, named_classifier in enumerate(self.named_classifiers):
            print( "name {} : {}".format(i, self.named_classifiers[named_classifier]) )

        return

    def fit( self, X_train, y_train, X_test ):
        #--------------------------------
        # １段目の k-fold CV での学習 & 推論
        #--------------------------------
        kf = StratifiedKFold( n_splits=self.n_splits, shuffle=True, random_state=self.seed )
        y_preds_train = np.zeros( (self.n_classifier, len(y_train)) )
        y_preds_test = np.zeros( (self.n_classifier, self.n_splits, len(X_test)) )
        #print( "y_preds_train.shape : ", y_preds_train.shape )
        #print( "y_preds_test.shape : ", y_preds_test.shape )

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
            self.fitted_classifiers = []
            for i, clf in enumerate( tqdm(self.classifiers, desc="fitting classifiers") ):
                if( self.clone ):
                    fitted_clf = clone(clf).fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )
                else:
                    fitted_clf = clf.fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

                self.fitted_classifiers.append( fitted_clf )

            #-------------------
            # 各モデルの推論処理
            #-------------------
            for i, clf in enumerate(self.fitted_classifiers):
                y_preds_train[i][valid_index] = clf.predict(X_valid_fold)
                y_preds_test[i][k] = clf.predict(X_test)

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
        if( self.second_classifiers != None ):
            kf = StratifiedKFold( n_splits=self.n_splits, shuffle=True, random_state=self.seed )
            y_preds_train = np.zeros( (self.n_second_classifier, len(y_train)) )
            y_preds_test = np.zeros( (self.n_second_classifier, self.n_splits, len(X_test)) )

            print( "[second_classifiers] X_train.shape : ", X_train.shape )
            print( "[second_classifiers] y_train.shape : ", y_train.shape )
            
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
                self.fitted_second_classifiers = []
                for i, clf in enumerate( tqdm(self.second_classifiers, desc="fitting second classifiers") ):
                    if( self.clone ):
                        fitted_clf = clone(clf).fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )
                    else:
                        fitted_clf = clf.fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

                    self.fitted_second_classifiers.append( fitted_clf )

                #-------------------
                # 各モデルの推論処理
                #-------------------
                for i, clf in enumerate(self.fitted_second_classifiers):
                    y_preds_train[i][valid_index] = clf.predict(X_valid_fold)
                    y_preds_test[i][k] = clf.predict(X_test)

                k += 1

            # テストデータに対する予測値の k-fold CV の平均をとる
            y_preds_test = np.mean( y_preds_test, axis=1 )
            #print( "y_preds_test.shape : ", y_preds_test.shape )

            # 各モデルの予想値をスタッキング
            y_preds_train_stack = y_preds_train[0]
            y_preds_test_stack = y_preds_test[0]
            for i in range(self.n_second_classifier - 1):
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
                clone(self.final_classifiers).fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )
            else:
                self.final_classifiers.fit( X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

            #-------------------
            # 各モデルの推論処理
            #-------------------
            y_preds_train[valid_index] = self.final_classifiers.predict(X_valid_fold)
            y_preds_test[k] = self.final_classifiers.predict(X_test)
            k += 1

        # 正解率の計算
        self.accuracy = (y_train == y_preds_train).sum()/len(y_preds_train)
        print( "[EnsembleStackingClassifier] accuracy [k-fold CV train vs valid] : {:0.5f}".format(self.accuracy) )

        # テストデータに対する予測値の平均をとる
        self.y_preds_test = np.mean( y_preds_test, axis=0 )
        self.y_preds_train = y_preds_train
        return self

    def predict( self, X_test ):
        return self.y_preds_test


    def predict_proba( self, X_test ):
        return self.y_preds_test