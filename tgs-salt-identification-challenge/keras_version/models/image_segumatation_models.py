import os
import numpy as np

# Keras
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, MaxPooling2D, concatenate, Dropout

class KerasUnet():
    def __init__( 
        self, n_in_channels = 3, n_out_channels = 1, n_fmaps = 64, 
        n_epoches = 10, batch_size = 32, batch_size_test = 256, lr = 0.001, beta1 = 0.5, beta2 = 0.999,
        use_valid = False, one_hot_encode = True, callbacks = None, use_datagen = False, datagen = None, debug = False
    ):
        self.model = None
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.use_valid = use_valid
        self.one_hot_encode = one_hot_encode
        self.callbacks = callbacks
        self.use_datagen = use_datagen
        self.datagen = datagen
        self.debug = debug
        self.evals_result = []

        # モデルの定義
        def conv_block( model, in_dim, out_dim ):
            model.add( Conv2D( filters = in_dim, kernel_size = (3,3), strides = 1, padding = "same", input_shape = (in_dim,) ) )
            model.add( BatchNormalization() )
            model.add( LeakyReLU(alpha=0.3) )
            return model

        def dconv_block( in_dim, out_dim ):
            return

        self.model = Sequential()
        self.model = conv_block(self.model, n_in_channels, n_fmaps )

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
        }
        return params

    def set_params(self, **params):
        self.n_epoches = params["n_epoches"]
        self.batch_size = params["batch_size"]
        return self

    def fit( self, X_train, y_train, X_valid = None, y_valid = None ):
        # 学習処理
        evals_result = {}
        if( self.use_datagen ):
            if( self.use_valid ):
                evals_result = self.model.fit_generator( 
                    self.datagen.flow( X_train, y_train, batch_size=self.batch_size ), 
                    epochs = self.n_epoches,
                    steps_per_epoch = math.ceil(len(X_train) / self.batch_size),
                    validation_data = ( X_valid, y_valid ),
                    shuffle = True, verbose = 1,
                    callbacks = self.callbacks,
                    workers = -1, use_multiprocessing = True,
                )
            else:
                evals_result = self.model.fit_generator( 
                    self.datagen.flow(X_train, y_train, batch_size=self.batch_size), 
                    epochs = self.n_epoches,
                    steps_per_epoch = math.ceil(len(X_train) / self.batch_size),
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

        self.evals_result = evals_result.history
        return self

    def evaluate( self, X_train, y_train ):
        evals_result = self.model.evaluate( X_train, y_train, batch_size = self.batch_size_test, verbose = 1, )
        return evals_result

    def predict(self, X_test):
        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        predicts = np.argmax(predicts, axis = 1)
        return predicts

    def predict_proba(self, X_test):
        # 推論処理
        predicts = self.model.predict( X_test, use_multiprocessing = True, verbose = 1 )
        #predicts = predicts[:,1] 
        return predicts

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