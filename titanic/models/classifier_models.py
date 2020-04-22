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

# XGBoost
import xgboost as xgb

# Keras
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.utils import to_categorical


class SklearnClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( self, model, debug = False ):
        self.model = model
        self.debug = debug
        return

    def load_params( self, file_path ):
        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
        return

    def fit( self, X_train, y_train, X_valid, y_valid ):
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
    def __init__( self, params_file_path = "parames/xgboost_classifier_default.yml", debug = False ):
        self.model = None
        self.debug = debug
        self.evals_results = []
        self.load_params( params_file_path )
        return

    def load_params( self, file_path ):
        if( self.debug ):
            print( "load parame file_path : ", file_path )

        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
            self.model_params = self.params["model"]["model_params"]
            self.train_params = self.params["model"]["train_params"]
        return

    def get_eval_results( self ):
        return self.evals_results

    def fit( self, X_train, y_train, X_valid, y_valid ):
        # XGBoost 用データセットに変換
        X_train_dmat = xgb.DMatrix(X_train, label=y_train)
        X_valid_dmat = xgb.DMatrix(X_valid, label=y_valid)

        # 学習処理
        evals_result = {}
        self.model = xgb.train(
            self.model_params, X_train_dmat, 
            num_boost_round = self.train_params["num_boost_round"],
            early_stopping_rounds = self.train_params["early_stopping_rounds"],
            evals = [ (X_train_dmat, 'train'), (X_valid_dmat, 'valid') ],
            evals_result = evals_result,
            verbose_eval = self.train_params["num_boost_round"] // 50,
        )

        self.evals_results.append(evals_result)
        return self
        
    def predict(self, X_test):
        # XGBoost 用データセットに変換
        X_test_dmat = xgb.DMatrix(X_test)

        # 推論処理
        predicts = self.model.predict(X_test_dmat)

        # ラベル値を argmax で 0 or 1 の離散値にする
        #predicts = np.argmax(predicts, axis = 1)
        """
        if( self.debug ):
            print( "[XBboost] predicts.shape : ", predicts.shape )
            #print( "[XBboost] predicts[0:10] : ", predicts[0:5] )
        """
        return predicts

    def predict_proba(self, X_test):
        # XGBoost 用データセットに変換
        X_test_dmat = xgb.DMatrix(X_test)

        # 推論処理
        predicts = self.model.predict(X_test_dmat)
        """
        if( self.debug ):
            print( "[XBboost] predicts.shape : ", predicts.shape )
            #print( "[XBboost] predicts[0:10] : ", predicts[0:5] )
        """
        return predicts


class KerasDNNClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( self, n_input_dim, n_fmaps = 64, n_classes = 2, n_epoches = 50, batch_size = 128, debug = False ):
        self.model = None
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.debug = debug

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

        self.model.add( keras.layers.Dense(n_fmaps*3) )
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

        return

    def fit( self, X_train, y_train, X_valid, y_valid ):
        # numpy 型に変換
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)

        # one-hot encode
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

        # 学習処理
        self.model.fit( 
            X_train, y_train, 
            epochs = self.n_epoches, batch_size = self.batch_size,
            validation_data = ( X_valid, y_valid ),
            shuffle = True, verbose = 1,
        )

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


class KerasResNetClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( 
        self, 
        n_channles,
        image_height = 224, image_width = 224,
        n_classes = 2, 
        n_epoches = 10, batch_size = 32,
        debug = False
    ):
        self.model = None
        self.n_channles = n_channles
        self.image_height = image_height
        self.image_width = image_width        
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.n_classes = n_classes

        self.debug = debug

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
        X_valid = scaler.transform(X_valid)

        # shape = [N, C] -> [N,H,W,C]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, 1, self.n_channles) )
        X_valid = np.reshape(X_valid, (X_valid.shape[0], 1, 1, self.n_channles) )
        X_train = np.concatenate( [X_train for i in range(self.image_height)], 1 )
        X_train = np.concatenate( [X_train for i in range(self.image_width)], 2 )
        X_valid = np.concatenate( [X_valid for i in range(self.image_height)], 1 )
        X_valid = np.concatenate( [X_valid for i in range(self.image_width)], 2 )

        # one-hot encode
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

        # 学習処理
        self.model.fit( 
            X_train, y_train, 
            epochs = self.n_epoches, batch_size = self.batch_size,
            validation_data = ( X_valid, y_valid ),
            shuffle = True, verbose = 1,
        )

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

