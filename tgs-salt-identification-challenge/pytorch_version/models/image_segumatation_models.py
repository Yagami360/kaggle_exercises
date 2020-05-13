import os
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns

# Keras
import keras
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, MaxPooling2D, Dropout, concatenate, Reshape
from keras.layers.core import Dense, Activation, RepeatVector

class KerasUnet():
    def __init__( 
        self, image_height = 128, image_width = 128, n_in_channels = 1, n_out_channels = 1, n_fmaps = 16, 
        n_epoches = 10, batch_size = 32, batch_size_test = 256, lr = 0.001, beta1 = 0.5, beta2 = 0.999,
        use_valid = False, callbacks = None, use_datagen = False, datagen = None, debug = False
    ):
        self.model = None
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels        
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.use_valid = use_valid
        self.callbacks = callbacks
        self.use_datagen = use_datagen
        self.datagen = datagen
        self.debug = debug
        self.evals_result = []

        # モデルの定義
        self.model = self.define_model( image_height, image_width, n_in_channels, n_out_channels, n_fmaps )

        # 損失関数と最適化アルゴリズムの設定
        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer = optimizers.Adam( lr = lr, beta_1 = beta1, beta_2 = beta2 ),
            metrics = ['accuracy']
        )

        if( self.debug ):
            self.model.summary()
        return

    def define_model( self, image_height = 128, image_width = 128, n_in_channels = 1, n_out_channels = 1, n_fmaps = 64 ):
        inputs = Input( shape=(image_height, image_width, n_in_channels) )

        """
        # conv block 1
        conv1 = Conv2D( filters = n_fmaps, kernel_size = (3,3), strides = 1, padding = "same" )(inputs)
        #conv1 = BatchNormalization()(conv1)
        conv1 = LeakyReLU(alpha=0.3)(conv1)
        conv1 = Conv2D( filters = n_fmaps, kernel_size = (3,3), strides = 1, padding = "same" )(conv1)
        #conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        # conv block 2
        conv2 = Conv2D( filters = n_fmaps*2, kernel_size = (3,3), strides = 1, padding = "same" )(pool1)
        #conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D( filters = n_fmaps*2, kernel_size = (3,3), strides = 1, padding = "same" )(conv2)
        #conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.3)(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        # conv block 3
        conv3 = Conv2D( filters = n_fmaps*4, kernel_size = (3,3), strides = 1, padding = "same" )(pool2)
        #conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.3)(conv3)
        conv3 = Conv2D( filters = n_fmaps*4, kernel_size = (3,3), strides = 1, padding = "same" )(conv3)
        #conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)

        # conv block 4
        conv4 = Conv2D( filters = n_fmaps*8, kernel_size = (3,3), strides = 1, padding = "same" )(pool3)
        #conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.3)(conv4)
        conv4 = Conv2D( filters = n_fmaps*8, kernel_size = (3,3), strides = 1, padding = "same" )(conv4)
        #conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)

        # bridge
        bridge = Conv2D(n_fmaps * 16, (3, 3), activation="relu", padding="same")(pool4)
        #bridge = BatchNormalization()(bridge)
        bridge = LeakyReLU(alpha=0.3)(bridge)
        bridge = Conv2D(n_fmaps * 16, (3, 3), activation="relu", padding="same")(bridge)
        #bridge = BatchNormalization()(bridge)

        # deconv 1 (upsampling & skip connection)
        dconv1 = Conv2DTranspose(n_fmaps * 8, (3, 3), strides=(2, 2), padding="same")(bridge)
        #dconv1 = BatchNormalization()(dconv1)
        dconv1 = LeakyReLU(alpha=0.3)(dconv1)

        concat1 = concatenate( [dconv1, conv4] )
        up1 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(concat1)
        #up1 = BatchNormalization()(up1)
        up1 = LeakyReLU(alpha=0.3)(up1)
        up1 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(up1)
        #up1 = BatchNormalization()(up1)

        # deconv 2 (upsampling & skip connection)
        dconv2 = Conv2DTranspose(n_fmaps * 4, (3, 3), strides=(2, 2), padding="same")(up1)
        #dconv2 = BatchNormalization()(dconv2)
        dconv2 = LeakyReLU(alpha=0.3)(dconv2)

        concat2 = concatenate( [dconv2, conv3] )
        up2 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(concat2)
        #up2 = BatchNormalization()(up2)
        up2 = LeakyReLU(alpha=0.3)(up2)
        up2 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(up2)
        #up2 = BatchNormalization()(up2)

        # deconv 3 (upsampling & skip connection)
        dconv3 = Conv2DTranspose(n_fmaps * 2, (3, 3), strides=(2, 2), padding="same")(up2)
        #dconv3 = BatchNormalization()(dconv3)
        dconv3 = LeakyReLU(alpha=0.3)(dconv3)

        concat3 = concatenate( [dconv3, conv2] )
        up3 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(concat3)
        #up3 = BatchNormalization()(up3)
        up3 = LeakyReLU(alpha=0.3)(up3)
        up3 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(up3)
        #up3 = BatchNormalization()(up3)

        # deconv 4 (upsampling & skip connection)
        dconv4 = Conv2DTranspose(n_fmaps, (3, 3), strides=(2, 2), padding="same")(up3)
        #dconv4 = BatchNormalization()(dconv4)
        dconv4 = LeakyReLU(alpha=0.3)(dconv4)

        concat4 = concatenate( [dconv4, conv1] )
        up4 = Conv2D(n_fmaps, (3, 3), activation="relu", padding="same")(concat4)
        #up4 = BatchNormalization()(up4)
        up4 = LeakyReLU(alpha=0.3)(up4)
        up4 = Conv2D(n_fmaps, (3, 3), activation="relu", padding="same")(up4)
        #up4 = BatchNormalization()(up4)

        # output layer
        outputs = Conv2D( n_out_channels, (1,1), padding="same", activation="sigmoid" )(up4)
        """

        # 128 -> 64
        conv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(inputs)
        conv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)

        # 64 -> 32
        conv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)

        # 32 -> 16
        conv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)

        # 16 -> 8
        conv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)

        # Middle
        convm = Conv2D(n_fmaps * 16, (3, 3), activation="relu", padding="same")(pool4)
        convm = Conv2D(n_fmaps * 16, (3, 3), activation="relu", padding="same")(convm)

        # 8 -> 16
        deconv4 = Conv2DTranspose(n_fmaps * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(uconv4)

        # 16 -> 32
        deconv3 = Conv2DTranspose(n_fmaps * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(uconv3)

        # 32 -> 64
        deconv2 = Conv2DTranspose(n_fmaps * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(uconv2)

        # 64 -> 128
        deconv1 = Conv2DTranspose(n_fmaps * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(uconv1)

        outputs = Conv2D(n_out_channels, (1,1), padding="same", activation="sigmoid")(uconv1)

        model = Model(inputs, outputs)
        return model

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
        return predicts

    def plot_loss(self, save_path):
        if( self.debug ):
            print( "self.evals_result.keys(): ", self.evals_result.keys() )

        # loss
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.plot(self.evals_result['loss'], label='train')
        axis.plot(self.evals_result['val_loss'], label='valid')

        plt.xlabel('epoches')
        plt.ylabel("loss")
        plt.xlim( [0,self.n_epoches+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )

        # accuracy
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.plot(self.evals_result['accuracy'], label='train')
        axis.plot(self.evals_result['val_accuracy'], label='valid')

        plt.xlabel('epoches')
        plt.ylabel("accuracy")
        plt.xlim( [0,self.n_epoches+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( save_path.split(".")[0] + "_accuracy.png", dpi = 300, bbox_inches = 'tight' )
        return


class KerasUnetWithDepth():
    def __init__( 
        self, image_height = 128, image_width = 128, n_in_channels = 1, n_out_channels = 1, n_fmaps = 16, 
        n_epoches = 10, batch_size = 32, batch_size_test = 256, lr = 0.001, beta1 = 0.5, beta2 = 0.999,
        use_valid = False, callbacks = None, use_datagen = False, datagen = None, debug = False
    ):
        self.model = None
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels        
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.batch_size_test = batch_size_test
        self.use_valid = use_valid
        self.callbacks = callbacks
        self.use_datagen = use_datagen
        self.datagen = datagen
        self.debug = debug
        self.evals_result = []

        # モデルの定義
        self.model = self.define_model( image_height, image_width, n_in_channels, n_out_channels, n_fmaps )

        # 損失関数と最適化アルゴリズムの設定
        self.model.compile(
            loss = 'binary_crossentropy',
            optimizer = optimizers.Adam( lr = lr, beta_1 = beta1, beta_2 = beta2 ),
            metrics = ['accuracy']
        )

        if( self.debug ):
            self.model.summary()
        return

    def define_model( self, image_height = 128, image_width = 128, n_in_channels = 1, n_out_channels = 1, n_fmaps = 64 ):
        inputs_img = Input( shape=(image_height, image_width, n_in_channels), name='img' )
        inputs_depth = Input((1, ), name='depth')

        # 128 -> 64
        conv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(inputs_img)
        conv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        pool1 = Dropout(0.25)(pool1)

        # 64 -> 32
        conv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(pool1)
        conv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)

        # 32 -> 16
        conv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(pool2)
        conv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)

        # 16 -> 8
        conv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(pool3)
        conv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)

        # depth ベクトルを concat
        depth = RepeatVector(8*8)(inputs_depth)         # shape 1 -> shape 64
        depth = Reshape((8, 8, 1))(depth)               # shape 64 -> shape = [None, 8, 8, 1]
        pool4_depth = concatenate([pool4, depth], -1)   # bottle neck 部分で concat

        # bottle neck
        convm = Conv2D(n_fmaps * 16, (3, 3), activation="relu", padding="same")(pool4_depth)
        convm = Conv2D(n_fmaps * 16, (3, 3), activation="relu", padding="same")(convm)

        # 8 -> 16
        deconv4 = Conv2DTranspose(n_fmaps * 8, (3, 3), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Dropout(0.5)(uconv4)
        uconv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(n_fmaps * 8, (3, 3), activation="relu", padding="same")(uconv4)

        # 16 -> 32
        deconv3 = Conv2DTranspose(n_fmaps * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Dropout(0.5)(uconv3)
        uconv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(n_fmaps * 4, (3, 3), activation="relu", padding="same")(uconv3)

        # 32 -> 64
        deconv2 = Conv2DTranspose(n_fmaps * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Dropout(0.5)(uconv2)
        uconv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(n_fmaps * 2, (3, 3), activation="relu", padding="same")(uconv2)

        # 64 -> 128
        deconv1 = Conv2DTranspose(n_fmaps * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Dropout(0.5)(uconv1)
        uconv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(n_fmaps * 1, (3, 3), activation="relu", padding="same")(uconv1)

        outputs = Conv2D(n_out_channels, (1,1), padding="same", activation="sigmoid")(uconv1)

        model = Model( inputs = [inputs_img, inputs_depth], outputs = [outputs] )
        return model

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
        return predicts

    def plot_loss(self, save_path):
        if( self.debug ):
            print( "self.evals_result.keys(): ", self.evals_result.keys() )

        # loss
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.plot(self.evals_result['loss'], label='train')
        axis.plot(self.evals_result['val_loss'], label='valid')

        plt.xlabel('epoches')
        plt.ylabel("loss")
        plt.xlim( [0,self.n_epoches+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( save_path, dpi = 300, bbox_inches = 'tight' )

        # accuracy
        fig = plt.figure()
        axis = fig.add_subplot(111)
        axis.plot(self.evals_result['accuracy'], label='train')
        axis.plot(self.evals_result['val_accuracy'], label='valid')

        plt.xlabel('epoches')
        plt.ylabel("accuracy")
        plt.xlim( [0,self.n_epoches+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( save_path.split(".png")[0] + "_accuracy.png", dpi = 300, bbox_inches = 'tight' )
        return