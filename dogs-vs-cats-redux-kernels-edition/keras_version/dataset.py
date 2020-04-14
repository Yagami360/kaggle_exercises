import os
import numpy as np
import random
import re
import math
import cv2

# keras
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras import backend as K

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class DogsVSCatsDataset(Sequential):
    def __init__(self, args, root_dir, datamode = "train", mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), enable_da = False, debug = False ):
        super(DogsVSCatsDataset, self).__init__()
        self.args = args
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.batch_size = args.batch_size
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
        """
        return math.ceil(len(self.image_names) / self.batch_size)

    def __getitem__(self, idx):
        print( "idx : ", idx )
        idx_start = idx * self.batch_size
        idx_last = idx_start + self.batch_size
        image_name = self.image_names[idx]

        # X_train
        X_train_batch = cv2.imread( os.path.join(self.dataset_dir, image_name) )
        X_train_batch = cv2.resize( X_train_batch, (self.image_height, self.image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]

        # y_train
        if( "cat." in image_name ):
            y_train_batch = np.zeros(1, dtype=np.uint8)
        else:
            y_train_batch = np.ones(1, dtype=np.uint8)

        return X_train_batch, y_train_batch

    def on_epoch_end(self):
        """
        1エポック分の処理が完了した際に実行される。
        属性で持っている（__getitem__関数実行後も残る）データなどの破棄処理や
        コールバックなど、必要な処理があれば記載する。
        """
        return
