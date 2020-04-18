import os
import numpy as np
import random
import re
import math
import cv2

# keras
import keras
from keras.utils import Sequence
from keras.utils import to_categorical

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

def load_dataset(
    root_dir, 
    datamode = "train", 
    image_height = 224, image_width = 224, n_classes = 2,
    n_samplings = -1,
    one_hot_encode = True,
):
    dataset_dir = os.path.join( root_dir, datamode )
    image_names = sorted( [f for f in os.listdir(dataset_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )
    image_names = image_names[0: min(n_samplings, len(image_names))]

    X_feature = np.zeros( (len(image_names), image_height, image_width, 3), dtype=np.uint8 )
    for i, name in enumerate(image_names):
        img = cv2.imread( os.path.join(dataset_dir,name) )
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        X_feature[i] = img

    if( one_hot_encode ):
        y_label = np.zeros( (len(image_names), n_classes), dtype=np.uint8 )
    else:
        y_label = np.zeros( (len(image_names), 1), dtype=np.uint8 )

    for i, name in enumerate(image_names):
        if( "cat." in name ):
            if( one_hot_encode ):
                y_label[i] = to_categorical( 0, n_classes )
            else:
                y_label[i] = 0
        else:
            if( one_hot_encode ):
                y_label[i] = to_categorical( 1, n_classes )
            else:
                y_label[i] = 1

    return X_feature, y_label


class DogsVSCatsDataGen(Sequence):
    def __init__(
        self, 
        args, 
        root_dir, 
        datamode = "train", 
        image_height = 224, image_width = 224, batch_size = 64, n_classes = 2,
        enable_da = False,
        debug = False
    ):
        super(DogsVSCatsDataGen, self).__init__()
        self.args = args
        self.datamode = datamode
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dataset_dir = os.path.join( root_dir, datamode )
        self.image_names = sorted( [f for f in os.listdir(self.dataset_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )
        self.debug = debug
        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])

        return

    def __len__(self):
        """
        __len__メソッドは 1epoch あたりのイテレーション回数。
        通常は、サンプル数をバッチサイズで割った値（の切り上げ）
        ここでは、エポック数ではなく step 数で計測させるため 1 を返す
        """
        if( self.datamode == "train" ):
            return 1
        else:
            return math.ceil(len(self.image_names) / self.batch_size)

    def __getitem__(self, idx):
        #print( "idx : ", idx )
        idx_start = idx * self.batch_size
        idx_last = idx_start + self.batch_size
        image_names_batch = self.image_names[idx_start:idx_last]
        if idx_start > len(self.image_names):
            idx_start = len(self.image_names)

        # X_feature
        X_feature_batch = np.zeros( (self.batch_size, self.image_height, self.image_width, 3), dtype=np.uint8 )   # shape = [N,H,W,C]
        for i, name in enumerate(image_names_batch):
            img = cv2.imread( os.path.join(self.dataset_dir, name) )
            img = cv2.resize( img, (self.image_height, self.image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
            X_feature_batch[i] = img

        #cv2.imwrite( "_debug/X_feature.png", X_feature_batch[0] )

        # y_label
        y_label_batch = np.zeros( (self.batch_size, self.n_classes), dtype=np.uint32 )
        for i, name in enumerate(image_names_batch):
            if( "cat." in name ):
                y_label_batch[i] = to_categorical( 0, self.n_classes )
            else:
                y_label_batch[i] = to_categorical( 1, self.n_classes )

        
        # 学習(fit_generatorメソッド)では説明変数と目的変数の両方、予測(predict_generatorメソッド)では説明変数のみ扱うため、それぞれ tarin と test で異なる戻り値を設定
        if( self.datamode == "train" ):
            return X_feature_batch, y_label_batch
        else:
            return X_feature_batch

    def on_epoch_end(self):
        """
        1エポック分の処理が完了した際に実行される。
        属性で持っている（__getitem__関数実行後も残る）データなどの破棄処理や
        コールバックなど、必要な処理があれば記載する。
        """
        return
