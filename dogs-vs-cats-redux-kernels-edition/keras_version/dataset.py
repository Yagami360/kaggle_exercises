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

class DogsVSCatsDataset(Sequence):
    def __init__(
        self, 
        args, 
        root_dir, 
        datamode = "train", 
        image_height = 224, image_width = 224, batch_size = 64, n_classes = 2,
        enable_da = False,
        debug = False
    ):
        super(DogsVSCatsDataset, self).__init__()
        self.args = args
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
        """
        return math.ceil(len(self.image_names) / self.batch_size)

    def __getitem__(self, idx):
        #print( "idx : ", idx )
        idx_start = idx * self.batch_size
        idx_last = idx_start + self.batch_size
        image_names_batch = self.image_names[idx_start:idx_last]
        if idx_start > len(self.image_names):
            idx_start = len(self.image_names)

        # X_train
        X_train_batch = np.zeros( (self.batch_size, self.image_height, self.image_width, 3), dtype=np.uint8 )   # shape = [N,H,W,C]
        for i, name in enumerate(image_names_batch):
            img = cv2.imread( os.path.join(self.dataset_dir, name) )
            img = cv2.resize( img, (self.image_height, self.image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
            X_train_batch[i] = img

        #cv2.imwrite( "_debug/X_train.png", X_train_batch[0] )

        # y_train
        y_train_batch = np.zeros( (self.batch_size, self.n_classes), dtype=np.uint32 )
        for i, name in enumerate(image_names_batch):
            if( "cat." in name ):
                y_train_batch[i] = to_categorical( 0, self.n_classes )
            else:
                y_train_batch[i] = to_categorical( 1, self.n_classes )

        return X_train_batch, y_train_batch

    def on_epoch_end(self):
        """
        1エポック分の処理が完了した際に実行される。
        属性で持っている（__getitem__関数実行後も残る）データなどの破棄処理や
        コールバックなど、必要な処理があれば記載する。
        """
        return
