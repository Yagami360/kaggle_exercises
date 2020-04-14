import os
import numpy as np
import cv2

# keras
import keras
from keras.models import Sequential
from keras.models import Model
#from keras.applications import ResNet50

class ResNet50(Model):
    def __init__( 
        self,
        image_height = 224,
        image_width = 224,
        n_channles = 3,
        n_classes = 2,
        pretrained = True,
        train_only_fc = False,
    ):
        super( ResNet50, self ).__init__()
        self.train_only_fc = train_only_fc
        if( pretrained ):
            self.resnet50 = keras.applications.ResNet50(
                weights = 'imagenet',   # 事前学習済みモデルを使用する
                input_shape = (image_height, image_width, n_channles),
                include_top = False     # 出力層を除外した pretrained model をインポート
            )
        else:
            self.resnet50 = keras.applications.ResNet50(
                weights = None,
                input_shape = (image_height, image_width, n_channles),
                include_top = False     # 出力層を除外した pretrained model をインポート
            )

        self.fc_layer = keras.layers.Dense(n_classes, activation='softmax')

        return

    def call( self, inputs ):
        output = self.resnet50(inputs)
        print( "output.shape: ", output.shape )
        output = self.fc_layer(output)
        return
