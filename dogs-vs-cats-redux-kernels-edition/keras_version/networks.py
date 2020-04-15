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
        #self.fc_layer = Sequential()
        #self.fc_layer.add( keras.layers.Flatten(input_shape=base_model.output_shape[1:]) )
        #self.fc_layer.add( keras.layers.Dense(n_classes, activation='softmax') )
        #self.finetuned_resnet50 = Model( input=base_model.input, output=self.fc_layer(base_model.output) )
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.Dense(n_classes, activation='softmax')(x)
        self.finetuned_resnet50 = Model(inputs=base_model.input, outputs=x)

        # 主力層のみ学習対象にする
        if( self.train_only_fc ):
            for layer in self.finetuned_resnet50.layers[:100]:
                layer.trainable = False
                print( layer )
                
        return

    def call( self, inputs ):
        # [To Do] 
        # inetuned_resnet50 を fit しているので、このメソッドが call されていない
        # このメソッドが call されるように変更したい。
        output = self.finetuned_resnet50(inputs)
        #print( "output.shape: ", output.shape )
        return
