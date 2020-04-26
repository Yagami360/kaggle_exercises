import os
import numpy as np
import yaml
from matplotlib import pyplot as plt
import seaborn as sns

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator                      # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import RegressorMixin                     # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.utils.estimator_checks import check_estimator

from sklearn.pipeline import _name_estimators
from sklearn.base import clone

# GDBT
import xgboost as xgb
import lightgbm as lgb
import catboost

# Keras
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.utils import to_categorical


class SklearnRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, model, use_valid = False, debug = False ):
        self.model = model
        self.use_valid = use_valid
        self.debug = debug
        self.model_params = self.get_params()
        self.feature_names = []
        if( self.debug ):
            print( "model_params :\n", self.model_params )

        return

    def load_params( self, file_path ):
        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
            self.model_params = self.params["model"]["model_params"]

        self.model.set_params(**self.model_params)
        return

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        self.feature_names = X_train.columns
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        predicts = self.model.predict(X_test)
        #predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        predicts = self.model.predict_proba(X_test)
        #predicts = predicts[:,1] 
        return predicts

    def plot_importance(self, save_path):
        feature_importance = self.model.feature_importances_
        print('Feature Importances:')
        for i, feature in enumerate(self.feature_names):
            print( '\t{0:5s} : {1:>.6f}'.format(feature, feature_importance[i]) )

        _, axis = plt.subplots(figsize=(8,12))
        axis.barh( range(len(feature_importance)), feature_importance, tick_label = self.feature_names )
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.grid()
        plt.tight_layout()
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )
        return

    def plot_loss(self, save_path):
        return

class XGBoostRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, model, train_type = "fit", use_valid = False, debug = False ):
        self.model = model
        self.train_type = train_type
        self.debug = debug
        self.use_valid = use_valid
        self.evals_results = []

        self.model_params = self.model.get_params()
        self.train_params = {
            "num_boost_round": 1000,            # 試行回数
            "early_stopping_rounds": 500,      # early stopping を行う繰り返し回数
        }
        if( self.debug ):
            print( "model_params :\n", self.model_params )

        return

    def load_params( self, file_path ):
        if( self.debug ):
            print( "load parame file_path : ", file_path )

        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
            self.model_params = self.params["model"]["model_params"]
            self.train_params = self.params["model"]["train_params"]

        #self.set_params(**self.model_params)
        if( self.train_type == "fit" ):
            self.model = xgb.XGBRegressor( **self.model_params )

        return

    """
    def get_params(self, deep=True):
        return self.model.get_params(deep)
    """

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def get_eval_results( self ):
        return self.evals_results

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        if( self.train_type == "fit" ):
            # 学習処理
            self.model.fit(X_train, y_train)

        else:
            # XGBoost 用データセットに変換
            X_train_dmat = xgb.DMatrix(X_train, label=y_train)
            if( self.use_valid ):
                X_valid_dmat = xgb.DMatrix(X_valid, label=y_valid)

            # 学習処理
            evals_result = {}
            if( self.use_valid ):
                self.model = xgb.train(
                    self.model_params, X_train_dmat, 
                    num_boost_round = self.train_params["num_boost_round"],
                    early_stopping_rounds = self.train_params["early_stopping_rounds"],
                    evals = [ (X_train_dmat, 'train'), (X_valid_dmat, 'valid') ],
                    evals_result = evals_result,
                    verbose_eval = self.train_params["num_boost_round"] // 50,
                )
                self.evals_results.append(evals_result)
            else:
                self.model = xgb.train(
                    self.model_params, X_train_dmat, 
                    num_boost_round = self.train_params["num_boost_round"],
                    early_stopping_rounds = self.train_params["early_stopping_rounds"],
                    evals_result = evals_result,
                    verbose_eval = self.train_params["num_boost_round"] // 50,
                )
                self.evals_results.append(evals_result)

        return self
        
    def predict(self, X_test):
        if( self.train_type == "fit" ):
            # 推論処理
            predicts = self.model.predict(X_test)
        else:
            # XGBoost 用データセットに変換
            X_test_dmat = xgb.DMatrix(X_test)

            # 推論処理
            predicts = self.model.predict(X_test_dmat)

        if( self.debug ):
            print( "[XGBoost] predicts.shape={}, predicts[0:5]={} ".format(predicts.shape, predicts[0:5]) )

        return predicts

    def predict_proba(self, X_test):
        if( self.train_type == "fit" ):
            predicts = self.model.predict_proba(X_test)
        else:
            # XGBoost 用データセットに変換
            X_test_dmat = xgb.DMatrix(X_test)

            # 推論処理
            # xgb.train() での retrun の XGBRegresser では predict_proba 使用不可
            predicts = self.model.predict(X_test_dmat)

        return predicts

    def plot_importance(self, save_path):
        _, axis = plt.subplots(figsize=(8,12))
        xgb.plot_importance(
            self.model,
            ax = axis,
            importance_type = 'gain',
            show_values = False
        )
        plt.tight_layout()
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )
        return

    def plot_loss(self, save_path):
        if( self.train_type == "train" and self.use_valid == True ):
            fig = plt.figure()
            axis = fig.add_subplot(111)
            for i, evals_result in enumerate(self.evals_results):
                axis.plot(evals_result['train'][self.model_params["eval_metric"]], label='train')
            for i, evals_result in enumerate(self.evals_results):
                axis.plot(evals_result['valid'][self.model_params["eval_metric"]], label='valid')

            plt.xlabel('iters')
            plt.ylabel(self.model_params["eval_metric"])
            plt.xlim( [0,self.train_params["num_boost_round"]+1] )
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )

        return


class LightGBMRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, model, train_type = "fit", use_valid = False, debug = False ):
        self.model = model
        self.train_type = train_type
        self.debug = debug
        self.use_valid = use_valid
        self.evals_results = []

        self.model_params = self.model.get_params()
        self.train_params = {
            "num_boost_round": 5000,            # 試行回数
            "early_stopping_rounds": 1000,      # early stopping を行う繰り返し回数
        }
        if( self.debug ):
            print( "model_params :\n", self.model_params )

        return

    def load_params( self, file_path ):
        if( self.debug ):
            print( "load parame file_path : ", file_path )

        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
            if( "model_params" in self.params["model"] ):
                self.model_params = self.params["model"]["model_params"]
            if( "train_params" in self.params["model"] ):
                self.train_params = self.params["model"]["train_params"]

        self.set_params(**self.model_params)
        return

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        if( self.train_type == "fit" ):
            # 学習処理
            self.model.fit(X_train, y_train)
        else:
            # LightGBM 用データセットに変換
            X_train_lgb = lgb.Dataset(X_train, label=y_train)
            if( self.use_valid ):
                X_valid_lgb = lgb.Dataset(X_valid, label=y_valid)

            # 学習処理
            evals_result = {}
            if( self.use_valid ):
                self.model = lgb.train(
                    self.model_params, X_train_lgb, 
                    num_boost_round = self.train_params["num_boost_round"],
                    early_stopping_rounds = self.train_params["early_stopping_rounds"],
                    valid_sets = [ X_train_lgb, X_valid_lgb ],
                    valid_names = ['train', 'valid'],
                    evals_result = evals_result,
                    verbose_eval = self.train_params["num_boost_round"] // 50,
                )
                self.evals_results.append(evals_result)
            else:
                self.model = lgb.train(
                    self.model_params, X_train_lgb, 
                    num_boost_round = self.train_params["num_boost_round"],
                    early_stopping_rounds = self.train_params["early_stopping_rounds"],
                    evals_result = evals_result,
                    verbose_eval = self.train_params["num_boost_round"] // 50,
                )
                self.evals_results.append(evals_result)

        return self
        
    def predict(self, X_test):
        if( self.train_type == "fit" ):
            # 推論処理（確定値が返る）
            predicts = self.model.predict(X_test)
        else:
            # 推論処理（確率値が返る）
            predicts = self.model.predict(X_test, num_iteration=self.model.best_iteration)

            # ラベル値を 0 or 1 の離散値にする
            predicts = np.where(predicts > 0.5, 1, 0)

        if( self.debug ):
            print( "[LightBGM] predicts.shape={}, predicts[0:5]={} ".format(predicts.shape, predicts[0:5]) )

        return predicts

    def predict_proba(self, X_test):
        if( self.train_type == "fit" ):
            predicts = self.model.predict_proba(X_test)
        else:
            # 推論処理
            # xgb.train() での retrun の XGBClassifier では predict_proba 使用不可。
            predicts = self.model.predict(X_test)

        return predicts

    def plot_importance(self, save_path):
        _, axis = plt.subplots(figsize=(8, 4))
        lgb.plot_importance(
            self.model,
            figsize=(8, 16),
            importance_type = 'gain',
            grid = True,
        )
        plt.tight_layout()
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )
        return

    def plot_loss(self, save_path):
        if( self.train_type == "train" and self.use_valid == True ):
            fig = plt.figure()
            axis = fig.add_subplot(111)
            for i, evals_result in enumerate(self.evals_results):
                axis.plot(evals_result['train'][self.model_params["metric"]], label='train')
            for i, evals_result in enumerate(self.evals_results):
                axis.plot(evals_result['valid'][self.model_params["metric"]], label='valid')

            plt.xlabel('iters')
            plt.ylabel(self.model_params["metric"])
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )

        return


class CatBoostRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, model, use_valid = False, debug = False ):
        self.model = model
        self.debug = debug
        self.use_valid = use_valid
        self.evals_results = []
        self.feature_names = []
        self.model_params = self.get_params(True)
        if( self.debug ):
            print( "model_params :\n", self.model_params )

        return

    def load_params( self, file_path ):
        if( self.debug ):
            print( "load parame file_path : ", file_path )

        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
            if( "model_params" in self.params["model"] ):
                self.model_params = self.params["model"]["model_params"]
            if( "train_params" in self.params["model"] ):
                self.train_params = self.params["model"]["train_params"]

        self.set_params(**self.model_params)
        return

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        self.feature_names = X_train.columns

        # 学習処理
        if( self.use_valid ):
            self.model.fit(
                X_train, y_train, eval_set = [(X_valid, y_valid)], 
                use_best_model = True,
                verbose_eval = 100,
            )
        else:
            self.model.fit(
                X_train, y_train, 
                use_best_model = True
            )            

        return self
        
    def predict(self, X_test):
        # 推論処理
        predicts = self.model.predict(X_test)

        # ラベル値を argmax で 0 or 1 の離散値にする
        #predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        predicts = self.model.predict_proba(X_test)
        return predicts

    def plot_importance(self, save_path):
        feature_importance = self.model.get_feature_importance()
        _, ax = plt.subplots(figsize=(8, 4))
        ax.barh( range(len(feature_importance)), feature_importance, tick_label = self.feature_names )
        plt.xlabel('importance')
        plt.ylabel('features')
        plt.grid()
        plt.tight_layout()
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )
        return

    def plot_loss(self, save_path):
        if( self.use_valid ):
            fig = plt.figure()
            axis = fig.add_subplot(111)

            evals_result = self.model.get_evals_result()
            axis.plot(evals_result['learn'][self.model_params["loss_function"]], label='train')
            axis.plot(evals_result['validation'][self.model_params["loss_function"]], label='valid')

            plt.xlabel('iters')
            plt.ylabel("Logloss")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )

        return