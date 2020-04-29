import os
import numpy as np
import yaml
from matplotlib import pyplot as plt
import seaborn as sns

# scikit-learn ライブラリ関連
from sklearn.base import BaseEstimator                      # 推定器 Estimator の上位クラス. get_params(), set_params() 関数が定義されている.
from sklearn.base import ClassifierMixin                    # 推定器 Estimator の上位クラス. score() 関数が定義されている.
from sklearn.utils.estimator_checks import check_estimator
from sklearn.pipeline import _name_estimators 
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

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


class SklearnImageClassifier( BaseEstimator, ClassifierMixin ):
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
        self.model.set_params(**params)
        return self

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        # shape = [N,H,W,C] -> [N, H*W*C] 
        X_train = X_train.reshape(X_train.shape[0],-1)
        if( self.use_valid ):
            X_valid = X_valid.reshape(X_valid.shape[0],-1)

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict(X_test)
        #predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict_proba(X_test)
        #predicts = predicts[:,1] 
        return predicts

    def plot_importance(self, save_path):
        return

    def plot_loss(self, save_path):
        return


class XGBoostImageClassifier( BaseEstimator, ClassifierMixin ):
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
        with open( file_path ) as f:
            self.params = yaml.safe_load(f)
            if( "model_params" in self.params["model"] ):
                self.model_params = self.params["model"]["model_params"]
            if( "train_params" in self.params["model"] ):
                self.train_params = self.params["model"]["train_params"]

        if( self.debug ):
            print( "load parame file_path : ", file_path )
            print( "model_params :\n", self.model_params )

        #self.set_params(**self.model_params)
        if( self.train_type == "fit" ):
            self.model = xgb.XGBClassifier( **self.model_params )

        return

    """
    def get_params(self, deep=True):
        return self.model.get_params(deep)
    """

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        # shape = [N,H,W,C] -> [N, H*W*C] 
        X_train = X_train.reshape(X_train.shape[0],-1)
        if( self.use_valid ):
            X_valid = X_valid.reshape(X_valid.shape[0],-1)

        if( self.train_type == "fit" ):
            # モデルの定義
            #self.model = xgb.XGBClassifier( **self.model_params )

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
        X_test = X_test.reshape(X_test.shape[0],-1)
        if( self.train_type == "fit" ):
            # 推論処理（確定値が返る）
            predicts = self.model.predict(X_test)
        else:
            # XGBoost 用データセットに変換
            X_test_dmat = xgb.DMatrix(X_test)

            # 推論処理（確率値が返る）
            predicts = self.model.predict(X_test_dmat)

            # ラベル値を 0 or 1 の離散値にする
            predicts = np.where(predicts > 0.5, 1, 0)

        if( self.debug ):
            print( "[XGBoost] predicts.shape={}, predicts[0:5]={} ".format(predicts.shape, predicts[0:5]) )

        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        if( self.train_type == "fit" ):
            predicts = self.model.predict_proba(X_test)
        else:
            # XGBoost 用データセットに変換
            X_test_dmat = xgb.DMatrix(X_test)

            # 推論処理
            # xgb.train() での retrun の XGBClassifier では predict_proba 使用不可。
            predicts = self.model.predict(X_test_dmat)

        return predicts

    def plot_importance(self, save_path ):
        _, ax = plt.subplots(figsize=(8, 4))
        xgb.plot_importance(
            self.model,
            ax = ax,
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


class LightGBMImageClassifier( BaseEstimator, ClassifierMixin ):
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
        # shape = [N,H,W,C] -> [N, H*W*C] 
        X_train = X_train.reshape(X_train.shape[0],-1)
        if( self.use_valid ):
            X_valid = X_valid.reshape(X_valid.shape[0],-1)

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
        X_test = X_test.reshape(X_test.shape[0],-1)
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
        X_test = X_test.reshape(X_test.shape[0],-1)
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
            figsize=(8, 4),
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


class CatBoostImageClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( self, model, use_valid = False, debug = False ):
        self.model = model
        self.debug = debug
        self.use_valid = use_valid
        self.evals_results = []
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
        # shape = [N,H,W,C] -> [N, H*W*C] 
        X_train = X_train.reshape(X_train.shape[0],-1)
        if( self.use_valid ):
            X_valid = X_valid.reshape(X_valid.shape[0],-1)

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
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict(X_test)
        predicts = predicts.squeeze()  # shape = [N,1] -> shape = [N]
        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict_proba(X_test)
        return predicts

    def plot_importance(self, save_path):
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


class KerasMLPImageClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( 
        self, n_input_dim, n_fmaps = 64, n_classes = 2, 
        n_epoches = 10, batch_size = 32, lr = 0.001, beta1 = 0.5, beta2 = 0.999,
        use_valid = False, one_hot_encode = True, callbacks = None, use_datagen = False, datagen = None, debug = False
    ):
        self.model = None
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.use_valid = use_valid
        self.one_hot_encode = one_hot_encode
        self.callbacks = callbacks
        self.use_datagen = use_datagen
        self.datagen = datagen
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
            optimizer = optimizers.Adam( lr = lr, beta_1 = beta1, beta_2 = beta2 ),
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
        # shape = [N,H,W,C] -> [N, H*W*C]
        X_train = X_train.reshape(X_train.shape[0],-1)
        if( self.use_valid ):
            X_valid = X_valid.reshape(X_valid.shape[0],-1)

        # one-hot encode
        if( self.one_hot_encode ):
            y_train = to_categorical(y_train)
            if( self.use_valid ):
                y_valid = to_categorical(y_valid)

        # 学習処理
        evals_result = {}
        if( self.use_datagen ):
            if( self.use_valid ):
                evals_result = self.model.fit_generator( 
                    self.datagen.flow( X_train, y_train, batch_size=self.batch_size ), 
                    epochs = self.n_epoches,
                    validation_data = ( X_valid, y_valid ),
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                    workers = -1, use_multiprocessing = True,
                )
            else:
                evals_result = self.model.fit_generator( 
                    self.datagen.flow(X_train, y_train, batch_size=self.batch_size), 
                    epochs = self.n_epoches,
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                    workers = -1, use_multiprocessing = True,
                )
        else:
            if( self.use_valid ):
                evals_result = self.model.fit( 
                    X_train, y_train, 
                    epochs = self.n_epoches, batch_size = self.batch_size,
                    validation_data = ( X_valid, y_valid ),
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                )
            else:
                evals_result = self.model.fit( 
                    X_train, y_train, 
                    epochs = self.n_epoches, batch_size = self.batch_size,
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                )

        self.evals_results.append( evals_result.history )
        return self

    def predict(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        X_test = X_test.reshape(X_test.shape[0],-1)
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        #predicts = predicts[:,1] 
        return predicts

    def plot_importance(self, save_path):
        return

    def plot_loss(self, save_path):
        if( self.debug ):
            print( "self.evals_results[0].keys(): ", self.evals_results[0].keys() )

        # loss
        fig = plt.figure()
        axis = fig.add_subplot(111)
        for i, evals_result in enumerate(self.evals_results):
            axis.plot(evals_result['loss'], label='train')
        for i, evals_result in enumerate(self.evals_results):
            axis.plot(evals_result['val_loss'], label='valid')

        plt.xlabel('iters')
        plt.ylabel("loss")
        plt.xlim( [0,self.n_epoches+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )

        # accuracy
        fig = plt.figure()
        axis = fig.add_subplot(111)
        for i, evals_result in enumerate(self.evals_results):
            axis.plot(evals_result['accuracy'], label='train')
        for i, evals_result in enumerate(self.evals_results):
            axis.plot(evals_result['val_accuracy'], label='valid')

        plt.xlabel('iters')
        plt.ylabel("accuracy")
        plt.xlim( [0,self.n_epoches+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( save_path.split(".")[0] + "_accuracy.png", dpi = 300, bbox_inches = 'tight' )
        return


class KerasResNet50ImageClassifier( BaseEstimator, ClassifierMixin ):
    def __init__( 
        self, 
        image_height = 224, image_width = 224, n_channles = 3,
        n_classes = 2, 
        n_epoches = 10, batch_size = 32, lr = 0.001, beta1 = 0.5, beta2 = 0.999,
        pretrained = True, train_only_fc = True,
        use_valid = True, one_hot_encode = True, callbacks = None, use_datagen = False, datagen = None,debug = False
    ):
        self.model = None
        self.n_channles = n_channles
        self.image_height = image_height
        self.image_width = image_width        
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.use_valid = use_valid
        self.one_hot_encode = one_hot_encode
        self.callbacks = callbacks
        self.use_datagen = use_datagen
        self.datagen = datagen
        self.debug = debug
        self.evals_results = []

        # モデルの定義
        if( pretrained ):
            base_model = keras.applications.ResNet50(
                weights = 'imagenet',   # 事前学習済みモデルを使用する
                input_shape = (image_height, image_width, n_channles),
                include_top = False     # 出力層を除外した pretrained model をインポート
            )
        else:
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

        # 主力層のみ学習対象にする
        if( train_only_fc ):
            for layer in self.model.layers[:100]:
                layer.trainable = False
                print( layer )

        # 損失関数と最適化アルゴリズムのせ設定
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizers.Adam( lr = lr, beta_1 = beta1, beta_2 = beta2 ),
            metrics = ['accuracy']
        )

        return

    def fit( self, X_train, y_train, X_valid, y_valid ):
        # one-hot encode
        if( self.one_hot_encode ):
            y_train = to_categorical(y_train)
            if( self.use_valid ):
                y_valid = to_categorical(y_valid)

        # 学習処理
        if( self.use_datagen ):
            if( self.use_valid ):
                evals_result = self.model.fit_generator( 
                    self.datagen.flow( X_train, y_train, batch_size=self.batch_size ), 
                    epochs = self.n_epoches,
                    validation_data = ( X_valid, y_valid ),
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                    workers = -1, use_multiprocessing = True,
                )
            else:
                evals_result = self.model.fit_generator( 
                    self.datagen.flow(X_train, y_train, batch_size=self.batch_size), 
                    epochs = self.n_epoches,
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                    workers = -1, use_multiprocessing = True,
                )
        else:
            if( self.use_valid ):
                evals_result = self.model.fit( 
                    X_train, y_train, 
                    epochs = self.n_epoches, batch_size = self.batch_size,
                    validation_data = ( X_valid, y_valid ),
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                )
            else:
                evals_result = self.model.fit( 
                    X_train, y_train, 
                    epochs = self.n_epoches, batch_size = self.batch_size,
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                )

        self.evals_results.append( evals_result )

        return self

    def predict(self, X_test):
        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        predicts = predicts[:,1] 
        #predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        predicts = predicts[:,1] 
        return predicts

    def plot_importance(self, save_path):
        return

    def plot_loss(self, save_path):
        return


