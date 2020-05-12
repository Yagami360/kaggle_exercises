import os
import numpy as np
import random
import pandas as pd
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
    dataset_dir, 
    image_height_org = 101, image_width_org = 101,
    image_height = 128, image_width = 128, n_channels = 1,
    n_samplings = -1,
):
    df_train = pd.read_csv( os.path.join(dataset_dir, "train.csv") )
    print( df_train.head() )

    df_depth = pd.read_csv( os.path.join(dataset_dir, "depths.csv") )
    print( df_depth.head() )

    # rls mask のない画像名をクレンジング
    pass

    # サンプリングする画像名リスト
    image_names_train = df_train["id"]
    image_names_train = image_names_train[0: min(n_samplings, len(image_names_train))]
    for i, name in enumerate(image_names_train):
        image_names_train[i] = name + ".png"

    # X_train
    X_train = np.zeros( (len(image_names_train), image_height, image_width, n_channels), dtype=np.float32 )
    for i, name in enumerate(image_names_train):
        img = cv2.imread( os.path.join( dataset_dir, "train", "images", name ), cv2.IMREAD_GRAYSCALE ) / 255    # 0.0f ~ 1.0f
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )                # shape = [H,W,C]
        X_train[i] = img.reshape( (image_height, image_width, n_channels))

    # y_train（マスク画像）
    y_train = np.zeros( (len(image_names_train), image_height, image_width, 1), dtype=np.float32 )
    for i, name in enumerate(image_names_train):
        img = cv2.imread( os.path.join( dataset_dir, "train", "masks", name ), cv2.IMREAD_GRAYSCALE ) / 255     # 0.0f ~ 1.0f
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_NEAREST )                 # shape = [H,W,C]
        y_train[i] = img.reshape( (image_height, image_width, 1))

    # X_test
    image_names_test = sorted( [f for f in os.listdir(os.path.join( dataset_dir, "test", "images")) if f.endswith(IMG_EXTENSIONS)] )
    image_names_test = image_names_test[0: min(n_samplings, len(image_names_test))]
    X_test = np.zeros( (len(image_names_test), image_height, image_width, n_channels), dtype=np.float32 )
    for i, name in enumerate(image_names_test):
        img = cv2.imread( os.path.join( dataset_dir, "test", "images", name ), cv2.IMREAD_GRAYSCALE ) / 255     # 0.0f ~ 1.0f
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )                # shape = [H,W,C]
        X_test[i] = img.reshape( (image_height, image_width, n_channels))

    # クラスラベルを追加（画像内で塩が含まれる割合のクラス）
    """
    def cov_to_class(val):    
        for i in range(0, 11):
            if val * 10 <= i :
                return i

    coverage = np.zeros( (len(image_names_train), image_height, image_width, n_channels), dtype=np.float32 )
    
    np.sum(y_train) / ( image_height_org * image_width_org )
    
    coverage_class = []
    cov_to_class(coverage)
    """

    return X_train, y_train, X_test, image_names_train, image_names_test

