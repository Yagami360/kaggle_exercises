import os
import numpy as np
import yaml

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator                      # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import RegressorMixin                     # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.utils.estimator_checks import check_estimator

from sklearn.pipeline import _name_estimators
from sklearn.base import clone

# XGBoost
import xgboost as xgb


class SklearnRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, model, use_valid = False, debug = False ):
        self.model = model
        self.use_valid = use_valid
        self.debug = debug
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
        self.model.set_params(params)
        return self

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
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


class XGBoostRegressor( BaseEstimator, RegressorMixin ):
    def __init__( self, model, train_type = "fit", use_valid = False, debug = False ):
        self.model = model
        self.train_type = train_type
        self.debug = debug
        self.use_valid = use_valid
        self.evals_results = []
        return

    def load_params( self, file_path ):
        if( self.debug ):
            print( "load parame file_path : ", file_path )

        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
            self.model_params = self.params["model"]["model_params"]
            self.train_params = self.params["model"]["train_params"]

        self.set_params(**self.model_params)
        return

    """
    def get_params(self, deep=True):
        return self.model.get_params(deep)
    """

    def set_params(self, **params):
        self.model.set_params(params)
        return self

    def get_eval_results( self ):
        return self.evals_results

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        if( self.train_type == "fit" ):
            # モデルの定義
            self.model = xgb.XGBRegressor(
                booster = self.model_params['booster'],
                objective = self.model_params['objective'],
                learning_rate = self.model_params['learning_rate'],
                n_estimators = self.model_params['n_estimators'],
                max_depth = self.model_params['max_depth'],
                min_child_weight = self.model_params['min_child_weight'],
                subsample = self.model_params['subsample'],
                colsample_bytree = self.model_params['colsample_bytree'],
                gamma = self.model_params['gamma'],
                alpha = self.model_params['alpha'],
                reg_lambda = self.model_params['reg_lambda'],
                random_state = self.model_params['random_state']
            )

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

        # ラベル値を argmax で 0 or 1 の離散値にする
        #predicts = np.argmax(predicts, axis = 1)
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


