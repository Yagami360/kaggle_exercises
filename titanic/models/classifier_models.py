import os
import numpy as np
import yaml

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator                      # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import ClassifierMixin                    # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import _name_estimators 
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

# GDB
import xgboost as xgb
import lightgbm as lgb
import catboost

# Keras
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.utils import to_categorical


class SklearnClassifier( BaseEstimator, ClassifierMixin ):
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


class XGBoostClassifier( BaseEstimator, ClassifierMixin ):
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
            if( "model_params" in self.params["model"] ):
                self.model_params = self.params["model"]["model_params"]
            if( "train_params" in self.params["model"] ):
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
            self.model = xgb.XGBClassifier(
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
            # xgb.train() での retrun の XGBClassifier では predict_proba 使用不可
            predicts = self.model.predict(X_test_dmat)

        return predicts


class CatBoostClassifier( BaseEstimator, ClassifierMixin ):
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

    def get_eval_results( self ):
        return self.evals_results

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        if( self.train_type == "fit" ):
            # 学習処理
            self.model.fit(X_train, y_train)
        else:
            pass

        return self
        
    def predict(self, X_test):
        if( self.train_type == "fit" ):
            # 推論処理
            predicts = self.model.predict(X_test)
        else:
            pass

        # ラベル値を argmax で 0 or 1 の離散値にする
        #predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        if( self.train_type == "fit" ):
            predicts = self.model.predict_proba(X_test)
        else:
            pass

        return predicts


class KerasMLPClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( self, n_input_dim, n_fmaps = 64, n_classes = 2, n_epoches = 50, batch_size = 128, use_valid = False, debug = False ):
        self.model = None
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.use_valid = use_valid
        self.debug = debug
        self.evals_results = []

        # モデルの定義
        self.model = Sequential()
        self.model.add( keras.layers.Dense(n_fmaps, input_shape = (n_input_dim,)) )
        self.model.add( keras.layers.BatchNormalization() )
        self.model.add( keras.layers.Activation("relu") )
        self.model.add( keras.layers.Dropout(0.25) )

        self.model.add( keras.layers.Dense(n_fmaps*2) )        
        self.model.add( keras.layers.BatchNormalization() )
        self.model.add( keras.layers.Activation("relu") )
        self.model.add( keras.layers.Dropout(0.25) )

        self.model.add( keras.layers.Dense(n_fmaps*4) )
        self.model.add( keras.layers.BatchNormalization() )
        self.model.add( keras.layers.Activation("relu") )
        self.model.add( keras.layers.Dropout(0.25) )

        self.model.add( keras.layers.Dense(n_classes) )
        self.model.add( keras.layers.Activation("softmax") )

        # 損失関数と最適化アルゴリズムのせ設定
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizers.Adam( lr = 0.001, beta_1 = 0.5, beta_2 = 0.999 ),
            metrics = ['accuracy']
        )

        if( self.debug ):
            self.model.summary()
        return

    def get_params(self, deep=True):
        params = {
            "n_epoches": self.n_epoches,
            "batch_size": self.batch_size,
            "n_classes": self.n_classes,
        }
        return params

    def set_params(self, **params):
        self.n_epoches = params["n_epoches"]
        self.batch_size = params["batch_size"]
        self.n_classes = params["n_classes"]
        return self

    
    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        # 正規化処理
        pass

        # numpy 型に変換
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        if( self.use_valid ):
            X_valid = scaler.transform(X_valid)

        # one-hot encode
        y_train = to_categorical(y_train)
        if( self.use_valid ):
            y_valid = to_categorical(y_valid)

        # 学習処理
        evals_result = {}
        if( self.use_valid ):
            evals_results = self.model.fit( 
                X_train, y_train, 
                epochs = self.n_epoches, batch_size = self.batch_size,
                validation_data = ( X_valid, y_valid ),
                shuffle = True, verbose = 1,
            )
        else:
            evals_results = self.model.fit( 
                X_train, y_train, 
                epochs = self.n_epoches, batch_size = self.batch_size,
                shuffle = True, verbose = 1,
            )

        self.evals_results.append( evals_result )
        return self

    def predict(self, X_test):
        # numpy 型に変換
        scaler = StandardScaler()
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        # numpy 型に変換
        scaler = StandardScaler()
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        #predicts = predicts[:,1] 
        return predicts


class KerasResNetClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( 
        self, 
        n_channles,
        image_height = 224, image_width = 224,
        n_classes = 2, 
        n_epoches = 10, batch_size = 32,
        use_valid = False,
        debug = False
    ):
        self.model = None
        self.n_channles = n_channles
        self.image_height = image_height
        self.image_width = image_width        
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.use_valid = use_valid
        self.debug = debug
        self.evals_results = []

        # モデルの定義
        base_model = keras.applications.ResNet50(
            weights = None,
            input_shape = (image_height, image_width, n_channles),
            include_top = False     # 出力層を除外した pretrained model をインポート
        )

        # 出力層を置き換える
        fc_layer = Sequential()
        fc_layer.add( keras.layers.Flatten(input_shape=base_model.output_shape[1:]) )
        fc_layer.add( keras.layers.Dense(n_classes, activation='softmax') )
        self.model = Model( input=base_model.input, output=fc_layer(base_model.output) )

        # 損失関数と最適化アルゴリズムのせ設定
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizers.Adam( lr = 0.001, beta_1 = 0.5, beta_2 = 0.999 ),
            metrics = ['accuracy']
        )

        return

    def fit( self, X_train, y_train, X_valid, y_valid ):
        # numpy 型に変換
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        if( self.use_valid ):
            X_valid = scaler.transform(X_valid)

        # shape = [N, C] -> [N,H,W,C]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, 1, self.n_channles) )
        X_train = np.concatenate( [X_train for i in range(self.image_height)], 1 )
        X_train = np.concatenate( [X_train for i in range(self.image_width)], 2 )
        if( self.use_valid ):
            X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, 1, self.n_channles) )
            X_valid = np.concatenate( [X_valid for i in range(self.image_height)], 1 )
            X_valid = np.concatenate( [X_valid for i in range(self.image_width)], 2 )

        # one-hot encode
        y_train = to_categorical(y_train)
        if( self.use_valid ):
            y_valid = to_categorical(y_valid)

        # 学習処理
        evals_result = {}
        if( self.use_valid ):
            evals_result = self.model.fit( 
                X_train, y_train, 
                epochs = self.n_epoches, batch_size = self.batch_size,
                validation_data = ( X_valid, y_valid ),
                shuffle = True, verbose = 1,
            )
        else:
            evals_result = self.model.fit( 
                X_train, y_train, 
                epochs = self.n_epoches, batch_size = self.batch_size,
                shuffle = True, verbose = 1,
            )

        self.evals_results.append( evals_result )

        return self

    def predict(self, X_test):
        # numpy 型に変換
        scaler = StandardScaler()
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        predicts = predicts[:,1] 
        #predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        # numpy 型に変換
        scaler = StandardScaler()
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)

        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        predicts = predicts[:,1] 
        return predicts

